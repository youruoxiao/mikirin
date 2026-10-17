import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('database_release', Path(__file__).resolve().parents[1] / 'scripts/database_release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class DatabaseReleaseTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.output = self.root / 'output'
        self.output.mkdir()

    def package(self):
        source = self.root / 'source.db'
        source.write_bytes(bytes(range(256)) * 10)
        package = release.pack('tiny', [(source, 'db/content.db')], self.output, 64, {})
        manifest = self.output / 'manifest.json'
        release.save_json(manifest, {'format': release.FORMAT, 'packages': [package]})
        return source, manifest, package

    def test_split_verify_and_restore_preserve_all_bytes(self):
        source, manifest, package = self.package()
        self.assertGreater(len(package['parts']), 2)
        release.verify(manifest)
        destination = self.root / 'restored'
        release.verify(manifest, destination)
        self.assertEqual(source.read_bytes(), (destination / 'db/content.db').read_bytes())
        with self.assertRaises(FileExistsError):
            release.verify(manifest, destination)

    def test_corrupt_part_rejected(self):
        _, manifest, package = self.package()
        (self.output / package['parts'][0]['name']).write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            release.verify(manifest)

    def test_wrong_inner_hash_rejected(self):
        _, manifest, package = self.package()
        package['files'][0]['sha256'] = '0' * 64
        release.save_json(manifest, {'format': release.FORMAT, 'packages': [package]})
        with self.assertRaisesRegex(ValueError, 'Archived file checksum'):
            release.verify(manifest)

    def test_restore_can_omit_undownloaded_backup_package(self):
        source, manifest, package = self.package()
        other = {'name': 'backup', 'parts': [{'name': 'missing.part', 'bytes': 1, 'sha256': '0'*64}],
                 'files': [{'path': 'db/backup.db', 'bytes': 1, 'sha256': '0'*64}]}
        release.save_json(manifest, {'format': release.FORMAT, 'packages': [package, other]})
        target = self.root / 'runtime-only'
        release.verify(manifest, target, ['tiny'])
        self.assertEqual(source.read_bytes(), (target / 'db/content.db').read_bytes())
        self.assertFalse((target / 'db/backup.db').exists())

    def test_no_overwrite_of_existing_parts(self):
        source, _, _ = self.package()
        with self.assertRaises(FileExistsError):
            release.pack('tiny', [(source, 'db/content.db')], self.output, 64, {})

    def test_paths_and_duplicate_files_rejected(self):
        for path in ('../outside', '/outside', 'D:/outside', 'a\\b', './a', 'a/../b'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                release.safe_relative(path)
        _, manifest, package = self.package()
        package['files'].append(package['files'][0])
        release.save_json(manifest, {'format': release.FORMAT, 'packages': [package]})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            release.load_manifest(manifest)

    def test_sqlite_snapshot_includes_committed_wal(self):
        source, target = self.root / 'source.db', self.root / 'copy.db'
        with closing(sqlite3.connect(source)) as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE build(id TEXT,status TEXT)')
            db.execute("INSERT INTO build VALUES('new','building')")
            db.commit()
            self.assertEqual(release.snapshot(source, target), [{'id': 'new', 'status': 'building'}])
        with closing(sqlite3.connect(target)) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM build').fetchone()[0], 1)
        with self.assertRaises(FileExistsError):
            release.snapshot(source, target)

    def test_upload_requires_verified_manifest(self):
        _, manifest, package = self.package()
        release.save_json(self.output / 'verified.json', {'manifest_sha256': 'wrong', 'files_checked': True})
        with patch.object(release, 'gh') as remote, self.assertRaises(ValueError):
            release.upload(manifest, 'owner/repo', 'test')
        remote.assert_not_called()

    def test_upload_skips_verified_assets_and_only_then_publishes(self):
        _, manifest, package = self.package()
        release.verify(manifest)
        for name in ('RESTORE.md', 'database_release.py'):
            (self.output / name).write_text('test fixture', encoding='utf-8')
        records = list(package['parts'])
        for path in (manifest, self.output / 'RESTORE.md', self.output / 'database_release.py'):
            records.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': release.digest(path)})
        assets = [{'name': r['name'], 'state': 'uploaded', 'size': r['bytes'],
                   'digest': 'sha256:' + r['sha256']} for r in records]
        calls = []

        def remote(*args, **kwargs):
            calls.append((args, kwargs))
            if args[1].endswith('/releases?per_page=100'):
                return [[{'id': 1, 'tag_name': 'test', 'draft': True}]]
            if '/assets?' in args[1]:
                return [assets]
            if args[1:3] == ('-X', 'PATCH'):
                return {'draft': False}
            self.fail(f'Unexpected request: {args}')

        with patch.object(release, 'gh', side_effect=remote):
            release.upload(manifest, 'owner/repo', 'test')
        self.assertEqual(sum(args[1:3] == ('-X', 'PATCH') for args, _ in calls), 1)
        calls.clear()
        with patch.object(release, 'gh', side_effect=remote):
            release.upload(manifest, 'owner/repo', 'test', stage_only=True)
        self.assertFalse(any(args[1:3] == ('-X', 'PATCH') for args, _ in calls))
        assets[0]['digest'] = 'sha256:' + '0' * 64
        calls.clear()
        with patch.object(release, 'gh', side_effect=remote), self.assertRaisesRegex(ValueError, 'differs'):
            release.upload(manifest, 'owner/repo', 'test')
        self.assertFalse(any(args[1:3] == ('-X', 'PATCH') for args, _ in calls))


if __name__ == '__main__':
    unittest.main()

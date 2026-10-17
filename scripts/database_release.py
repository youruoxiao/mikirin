"""Explicit content-database release packaging; no personal DB or credentials.

prepare writes consistent SQLite snapshots, gzip/tar parts, and a hash manifest.
verify streams the parts and checks every archived file. restore requires a NEW
directory. upload uses gh, verifies server SHA-256, and publishes only when all
assets have been verified. No production pointer or database is modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import subprocess
import tarfile
import threading
import time

FORMAT = 'mikirin-data-release-v1'
BLOCK = 4 * 1024 * 1024
PRINT_LOCK = threading.Lock()


def log(message):
    with PRINT_LOCK:
        print(message, flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(BLOCK):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe archive path: {name}')
    if path.as_posix() != name or any(p.startswith('.') for p in path.parts):
        raise ValueError(f'Noncanonical archive path: {name}')
    reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
    if any(p != p.rstrip(' .') or p.split('.')[0].upper() in reserved for p in path.parts):
        raise ValueError(f'Unsafe Windows archive path: {name}')
    return path


class SplitWriter:
    def __init__(self, directory, prefix, limit):
        self.directory, self.prefix, self.limit = Path(directory), prefix, limit
        if limit <= 0 or limit >= 2 * 1024**3:
            raise ValueError('Part size must be positive and under 2 GiB')
        self.parts, self.stream, self.size, self.hash = [], None, 0, None

    def write(self, data):
        original = len(data)
        data = memoryview(data)
        while data:
            if self.stream is None:
                self.name = f'{self.prefix}.tar.gz.part{len(self.parts)+1:04d}'
                self.stream = (self.directory / self.name).open('xb')
                self.size, self.hash = 0, hashlib.sha256()
            count = min(len(data), self.limit - self.size)
            block = data[:count]
            self.stream.write(block)
            self.hash.update(block)
            self.size += count
            data = data[count:]
            if self.size == self.limit:
                self._finish()
        return original

    def flush(self):
        if self.stream:
            self.stream.flush()

    def _finish(self):
        if self.stream:
            self.stream.close()
            self.parts.append({'name': self.name, 'bytes': self.size, 'sha256': self.hash.hexdigest()})
            log(f'part ready: {self.name} ({self.size:,} bytes)')
            self.stream = None

    def close(self):
        self._finish()


class PartsReader(io.RawIOBase):
    def __init__(self, directory, parts):
        self.directory, self.parts = Path(directory), iter(parts)
        self.stream = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.stream is None:
                item = next(self.parts, None)
                if item is None:
                    return 0
                self.stream = (self.directory / item['name']).open('rb')
            size = self.stream.readinto(buffer)
            if size:
                return size
            self.stream.close()
            self.stream = None

    def close(self):
        if self.stream:
            self.stream.close()
        super().close()


def snapshot(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    log(f'snapshot: {source.name}')
    with closing(sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)) as src:
        with closing(sqlite3.connect(target)) as dst:
            src.backup(dst, pages=16384)
            log(f'integrity check: {source.name}')
            checks = [row[0] for row in dst.execute('PRAGMA integrity_check')]
            if checks != ['ok']:
                raise ValueError(f'{source.name}: integrity check failed: {checks[:5]}')
            builds = [dict(zip(('id', 'status'), row)) for row in dst.execute('SELECT id,status FROM build ORDER BY id')]
    log(f'snapshot verified: {source.name}')
    return builds


def pack(name, files, output, part_size, extra):
    writer = SplitWriter(output, name, part_size)
    records = []
    try:
        with gzip.GzipFile(fileobj=writer, mode='wb', compresslevel=1, mtime=0, filename='') as gz:
            with tarfile.open(fileobj=gz, mode='w|') as archive:
                for source, member in files:
                    safe_relative(member)
                    before = source.stat()
                    h = hashlib.sha256()

                    class HashReader:
                        def read(self, size=-1):
                            data = stream.read(size)
                            h.update(data)
                            return data

                    info = tarfile.TarInfo(member)
                    info.size, info.mode, info.mtime = before.st_size, 0o644, 0
                    log(f'compress: {name}/{member} ({info.size:,} bytes)')
                    with source.open('rb') as stream:
                        archive.addfile(info, HashReader())
                    after = source.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError(f'Source changed during compression: {member}')
                    records.append({'path': member, 'bytes': info.size, 'sha256': h.hexdigest()})
    finally:
        writer.close()
    result = {'name': name, 'files': records, 'parts': writer.parts, **extra}
    save_json(Path(output) / f'{name}.package.json', result)
    return result


def prepare(root, output, part_mib=256):
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    pointer = json.loads((root / 'db/current.json').read_text(encoding='utf-8'))
    if pointer['database'] != 'content.v2026.08.01-a36.db':
        raise ValueError('Published database changed; explicitly review release scope first')
    vector_pointer = json.loads((root / 'data/vector_index/current.json').read_text(encoding='utf-8'))
    relative = safe_relative(vector_pointer['directory'])
    bundle = root / 'data/vector_index' / relative
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['source_database']['path'] != pointer['database']:
        raise ValueError('Vector bundle does not match the content database')
    if manifest['bundle_fingerprint'] != vector_pointer['bundle_fingerprint']:
        raise ValueError('Vector pointer fingerprint mismatch')
    snapshots = output / 'snapshots'
    snapshots.mkdir()
    # These pointers are release payloads, not changes to the live application.
    save_json(snapshots / 'content-current.json', pointer)
    save_json(snapshots / 'vector-current.json', vector_pointer)

    def content_job(name, filename, published):
        source, target = root / 'db' / filename, snapshots / filename
        builds = snapshot(source, target)
        files = [(target, 'db/' + filename)]
        if published:
            files.append((snapshots / 'content-current.json', 'db/current.json'))
        return pack(name, files, output, part_mib * 1024**2,
                    {'role': 'published-runtime' if published else 'unfinished-build-backup',
                     'builds': builds, 'sqlite_integrity': 'ok'})

    def vector_job():
        files = [(snapshots / 'vector-current.json', 'data/vector_index/current.json')]
        for source in sorted(bundle.rglob('*')):
            if source.is_symlink():
                raise ValueError('Bundle symlinks are not allowed')
            if source.is_file():
                if source.name not in {'manifest.json', 'header.json', 'meta.db', 'norms.bin', 'signatures.bin', 'vectors.int8'}:
                    raise ValueError(f'Unexpected bundle file: {source.name}')
                files.append((source, source.relative_to(root).as_posix()))
        return pack('runtime-vectors', files, output, part_mib * 1024**2,
                    {'role': 'published-runtime-vectors', 'bundle_fingerprint': manifest['bundle_fingerprint'],
                     'source_database': manifest['source_database']})

    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(content_job, 'runtime-content', pointer['database'], True),
                 pool.submit(content_job, 'p1-backup', 'content.v2026.08.02-p1.db', False),
                 pool.submit(vector_job)]
        packages = [task.result() for task in as_completed(tasks)]
    release = {'format': FORMAT, 'packages': sorted(packages, key=lambda p: p['name']),
               'excluded': ['user.db', 'personal sessions', 'credentials', 'build/LLM caches', 'obsolete vector bundles'],
               'rights_note': 'Publicly accessible source content; no blanket public-domain or CC0 claim. See source-specific terms and provenance.'}
    save_json(output / 'manifest.json', release)
    log('All packages prepared. Run verify before upload.')


def load_manifest(path):
    path = Path(path)
    result = json.loads(path.read_text(encoding='utf-8'))
    if result.get('format') != FORMAT or not result.get('packages'):
        raise ValueError('Unsupported manifest')
    names, members = set(), set()
    for package in result['packages']:
        for part in package['parts']:
            safe_relative(part['name'])
            if '/' in part['name'] or part['name'] in names or not 0 < part['bytes'] < 2 * 1024**3:
                raise ValueError('Invalid or duplicate part')
            names.add(part['name'])
        for file in package['files']:
            safe_relative(file['path'])
            if file['path'] in members or file['bytes'] < 0:
                raise ValueError('Invalid or duplicate archived file')
            members.add(file['path'])
    return result


def verify_package(directory, package, destination=None):
    for part in package['parts']:
        path = Path(directory) / part['name']
        if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
            raise ValueError(f'Part checksum mismatch: {part["name"]}')
    expected = {f['path']: f for f in package['files']}
    seen = set()
    with io.BufferedReader(PartsReader(directory, package['parts']), BLOCK) as source:
        with gzip.GzipFile(fileobj=source) as gz:
            with tarfile.open(fileobj=gz, mode='r|') as archive:
                for member in archive:
                    safe_relative(member.name)
                    if not member.isfile() or member.name not in expected or member.name in seen:
                        raise ValueError(f'Unexpected archive member: {member.name}')
                    record = expected[member.name]
                    if record['bytes'] != member.size:
                        raise ValueError('Archive size mismatch')
                    seen.add(member.name)
                    h, length = hashlib.sha256(), 0
                    output = None
                    if destination is not None:
                        target = Path(destination) / member.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        output = target.open('xb')
                    try:
                        with archive.extractfile(member) as stream:
                            while block := stream.read(BLOCK):
                                h.update(block)
                                length += len(block)
                                if output:
                                    output.write(block)
                    finally:
                        if output:
                            output.close()
                    if length != record['bytes'] or h.hexdigest() != record['sha256']:
                        raise ValueError(f'Archived file checksum mismatch: {member.name}')
            # Force gzip CRC/trailer verification even after tar end-of-archive.
            while gz.read(BLOCK):
                pass
    if seen != expected.keys():
        raise ValueError('Archive is missing files')
    log(f'archive verified: {package["name"]}')


def verify(path, destination=None, packages=None):
    path = Path(path)
    manifest = load_manifest(path)
    selected = manifest['packages']
    if packages:
        if destination is None or set(packages) - {p['name'] for p in selected}:
            raise ValueError('Package selection is allowed only for restore and must name existing packages')
        selected = [p for p in selected if p['name'] in packages]
    if destination is not None:
        Path(destination).mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(verify_package, path.parent, p, destination) for p in selected]
        for task in as_completed(tasks):
            task.result()
    if destination is None:
        save_json(path.parent / 'verified.json', {'manifest_sha256': digest(path), 'files_checked': True})


def gh(*args, payload=None, timeout=120):
    result = subprocess.run(['gh', *args], input=None if payload is None else json.dumps(payload),
                            text=True, encoding='utf-8', capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout[:500])
    return json.loads(result.stdout) if result.stdout.strip().startswith(('{', '[')) else result.stdout.strip()


def upload(path, repo, tag, workers=3):
    path = Path(path).resolve()
    manifest = load_manifest(path)
    proof = json.loads((path.parent / 'verified.json').read_text(encoding='utf-8'))
    if proof.get('manifest_sha256') != digest(path) or proof.get('files_checked') is not True:
        raise ValueError('Manifest is not locally verified')
    # GitHub's get-by-tag endpoint may omit drafts without an existing git tag.
    pages = gh('api', f'repos/{repo}/releases?per_page=100', '--paginate', '--slurp')
    release = next((r for page in pages for r in page if r['tag_name'] == tag), None)
    if release is None:
        raise ValueError('Create the intended draft release first')
    release_id = release['id']

    def remote_assets():
        return gh('api', f'repos/{repo}/releases/{release_id}/assets?per_page=100', '--paginate', '--slurp')

    def find_remote(name):
        return next((a for page in remote_assets() for a in page if a['name'] == name), None)

    def send(record):
        file = path.parent / record['name']
        if file.stat().st_size != record['bytes'] or digest(file) != record['sha256']:
            raise ValueError(f'Local upload file changed: {file.name}')
        for attempt in range(1, 5):
            asset = find_remote(file.name)
            if asset is not None:
                if asset.get('state') == 'uploaded' and asset.get('size') == record['bytes'] and asset.get('digest') == 'sha256:' + record['sha256']:
                    log(f'remote verified: {file.name}')
                    return
                raise ValueError(f'Remote asset differs/incomplete; no overwrite performed: {file.name}')
            log(f'upload ({attempt}/4): {file.name}')
            try:
                gh('release', 'upload', tag, str(file), '--repo', repo, timeout=900)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                log(f'upload retry needed: {file.name}: {type(error).__name__}')
                # Only clean an empty failed placeholder created by this exact
                # attempt (the name was absent immediately before upload).
                partial = find_remote(file.name)
                if partial and partial.get('state') == 'starter' and partial.get('size') == 0:
                    gh('api', '-X', 'DELETE', f'repos/{repo}/releases/assets/{partial["id"]}')
                    log(f'removed empty failed upload placeholder: {file.name}')
                if attempt == 4:
                    raise
                time.sleep(attempt * 5)
        asset = find_remote(file.name)
        if not asset or asset.get('digest') != 'sha256:' + record['sha256']:
            raise ValueError(f'Upload could not be verified: {file.name}')

    assets = [part for package in manifest['packages'] for part in package['parts']]
    for file in (path, path.parent / 'RESTORE.md', path.parent / 'database_release.py'):
        assets.append({'name': file.name, 'bytes': file.stat().st_size, 'sha256': digest(file)})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(send, record) for record in assets]
        for task in as_completed(tasks):
            task.result()
    # Final inventory audit before making the draft visible.
    inventory = {a['name']: a for page in remote_assets() for a in page}
    for record in assets:
        asset = inventory.get(record['name'], {})
        if asset.get('state') != 'uploaded' or asset.get('size') != record['bytes'] or asset.get('digest') != 'sha256:' + record['sha256']:
            raise ValueError(f'Final server checksum mismatch: {record["name"]}')
    gh('api', '-X', 'PATCH', f'repos/{repo}/releases/{release_id}', '--input', '-',
       payload={'draft': False, 'prerelease': True, 'make_latest': 'false'})
    log(f'Published and verified {len(assets)} assets: https://github.com/{repo}/releases/tag/{tag}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--part-mib', type=int, default=256)
    for command in ('verify', 'restore', 'upload'):
        p = sub.add_parser(command)
        p.add_argument('manifest', type=Path)
        if command == 'restore':
            p.add_argument('--destination', type=Path, required=True)
            p.add_argument('--packages', nargs='+', help='Restore selected packages only (default: all)')
        if command == 'upload':
            p.add_argument('--repo', required=True)
            p.add_argument('--tag', required=True)
            p.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.root, args.output, args.part_mib)
    elif args.command == 'upload':
        upload(args.manifest, args.repo, args.tag, args.workers)
    else:
        verify(args.manifest, getattr(args, 'destination', None), getattr(args, 'packages', None))


if __name__ == '__main__':
    main()

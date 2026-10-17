"""Offline export safety tests; all writes are confined to temporary fixtures."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_source.py"
SPEC = importlib.util.spec_from_file_location("export_source", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class ExportSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "project"
        self.root.mkdir()
        self.write("README.md", "# Test project\n")

    def write(self, name: str, content: str = "source\n") -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def symlink(self, link: Path, target: Path, *, directory: bool = False) -> None:
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f"Symlink creation unavailable: {error}")

    def test_exact_allowlist_and_secret_exclusion(self) -> None:
        allowed = {
            "README.md", "pyproject.toml", "THIRD_PARTY_NOTICES.md", "LICENSE", ".env.example",
            ".gitignore", "run.ps1", "src/mikirin/__init__.py", "src/mikirin/cache.py",
            "src/mikirin/nested/module.py", "src/mikirin/prompts/system.md",
            "tests/test_example.py", "tests/nested/test_more.py", "scripts/export_source.py",
            "scripts/nested/task.py", "scripts/README.md", "docs/nested/guide.md",
            "agent/SOUL.md", ".github/workflows/ci.yml", ".github/dependabot.yaml",
            "src/mikirin/resources/benchmarks/retrieval-60.json",
        }
        excluded = {
            ".env", ".env.production", ".env.example.bak", ".git/config",
            ".git/hooks/tool.py", "random.py", "README.md.bak", "README.private.md",
            "credentials.json", "id_rsa", "server.db", "server.db-wal",
            "data/manual/README.md", "data/vector_index/meta.db", "data/tool.py",
            "work/tool.py", "work/README.md", "dist/old.zip", "db/meta.db",
            "src/other/module.py", "src/mikirin/native.so", "src/mikirin/module.pyc",
            "src/mikirin/.env", "src/mikirin/.env.example", "src/mikirin/secrets.py",
            "src/mikirin/__pycache__/hidden.py", "src/mikirin/build/generated.py",
            "src/mikirin/data/README.md", "src/mikirin/work/worker.py",
            "src/mikirin/app.egg-info/README.md", "src/mikirin/credentials/key.md",
            "src/mikirin/.hidden/code.py", "tests/fixtures/data/payload.py",
            "tests/cache/generated.py", "tests/.env.test", "scripts/task_local.py",
            "scripts/NOTES.md", "scripts/secrets.py", "docs/credentials.md",
            "docs/cache/note.md", "agent/.private.md", ".github/.env.yml",
            ".github/secrets.yaml", ".github/workflows/ci.yml.bak", ".github/readme.md",
            "src/mikirin/resources/runtime.json",
            "src/mikirin/resources/benchmarks/labels.private.json",
        }
        for name in allowed:
            self.write(name, f"allowed: {name}\n")
        for name in excluded:
            self.write(name, "DO_NOT_EXPORT_SECRET\n")
        output = exporter.export_source(self.root)
        self.assertEqual(output, self.root / "dist/mikirin-source.zip")
        with zipfile.ZipFile(output) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.namelist(), sorted(allowed))
            for name in archive.namelist():
                self.assertEqual(archive.read(name), (self.root / name).read_bytes())
                self.assertNotIn(b"DO_NOT_EXPORT_SECRET", archive.read(name))

    def test_repeat_export_is_byte_identical_and_has_normalized_metadata(self) -> None:
        source = self.write("src/mikirin/main.py", "print('hello')\n")
        output = exporter.export_source(self.root)
        first = output.read_bytes()
        os.utime(source, (1_700_000_000, 1_700_000_000))
        self.assertEqual(exporter.export_source(self.root).read_bytes(), first)
        self.assertEqual(exporter.export_source(self.root, Path("other.zip")).read_bytes(), first)
        with zipfile.ZipFile(output) as archive:
            for info in archive.infolist():
                self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
                self.assertEqual(info.external_attr >> 16, stat.S_IFREG | 0o644)

    def test_source_and_future_source_outputs_are_rejected_without_changes(self) -> None:
        source = self.write("src/mikirin/main.py", "keep me\n")
        for name in (
            "README.md", ".env.example", ".gitignore", "src/mikirin/main.py",
            "src/mikirin/new.py", "tests/new.py", "docs/new.md", ".github/new.yaml",
            "scripts/README.md", "src/mikirin/archive.zip", "tests/archive.zip",
            "agent/archive.zip", "DOCS/archive.zip", "README.md/archive.zip",
            ".git/archive.zip",
        ):
            with self.subTest(name=name), self.assertRaises((ValueError, OSError)):
                exporter.export_source(self.root, Path(name))
        self.assertEqual(source.read_text(encoding="utf-8"), "keep me\n")
        self.assertFalse((self.root / "DOCS").exists())

    def test_non_zip_and_existing_directory_outputs_are_rejected(self) -> None:
        env = self.write(".env", "keep secret\n")
        folder = self.root / "folder.zip"
        folder.mkdir()
        for path in (env, folder, self.root / "new.txt"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                exporter.export_source(self.root, path)
        self.assertEqual(env.read_text(encoding="utf-8"), "keep secret\n")

    def test_failed_export_preserves_existing_archive_and_cleans_temporary_file(self) -> None:
        output = exporter.export_source(self.root)
        original = output.read_bytes()
        with patch.object(exporter, "_read_source", side_effect=OSError("read failed")):
            with self.assertRaises(OSError):
                exporter.export_source(self.root)
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(list(output.parent.iterdir()), [output])

    def test_absolute_external_output_is_supported(self) -> None:
        output = self.base / "release/source.zip"
        self.assertEqual(exporter.export_source(self.root, output), output)
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(archive.namelist(), ["README.md"])

    def test_empty_root_is_rejected_without_creating_output(self) -> None:
        (self.root / "README.md").unlink()
        with self.assertRaises(ValueError):
            exporter.export_source(self.root)
        self.assertFalse((self.root / "dist").exists())

    def test_source_replacement_during_read_is_detected(self) -> None:
        source = self.root / "README.md"
        replacement = self.write("replacement.txt", "changed\n")
        original_open = os.open

        def replace_before_open(path, flags):
            os.replace(replacement, source)
            return original_open(path, flags)

        with patch.object(exporter.os, "open", side_effect=replace_before_open):
            with self.assertRaisesRegex(ValueError, "Source changed"):
                exporter._read_source(source)

    def test_excluded_symlink_is_not_traversed(self) -> None:
        self.symlink(self.root / "data", self.base, directory=True)
        with zipfile.ZipFile(exporter.export_source(self.root)) as archive:
            self.assertEqual(archive.namelist(), ["README.md"])

    def test_linked_project_root_is_rejected(self) -> None:
        link = self.base / "project-link"
        self.symlink(link, self.root, directory=True)
        with self.assertRaises(ValueError):
            exporter.export_source(link)

    def test_file_symlink_to_secret_is_rejected(self) -> None:
        secret = self.write(".env", "secret\n")
        self.symlink(self.root / "src/mikirin/leak.py", secret)
        with self.assertRaises(ValueError):
            exporter.export_source(self.root)
        self.assertFalse((self.root / "dist").exists())

    def test_external_directory_symlink_is_rejected(self) -> None:
        external = self.base / "outside"
        external.mkdir()
        (external / "leak.md").write_text("secret", encoding="utf-8")
        self.symlink(self.root / "docs", external, directory=True)
        with self.assertRaises(ValueError):
            exporter.export_source(self.root)

    def test_dangling_allowlisted_symlink_is_rejected(self) -> None:
        self.symlink(self.root / "run.ps1", self.base / "missing")
        with self.assertRaises(ValueError):
            exporter.export_source(self.root)

    def test_output_symlink_is_rejected_and_target_is_unchanged(self) -> None:
        source = self.root / "README.md"
        self.symlink(self.root / "export.zip", source)
        with self.assertRaises(ValueError):
            exporter.export_source(self.root, Path("export.zip"))
        self.assertEqual(source.read_text(encoding="utf-8"), "# Test project\n")

    def test_output_parent_symlink_is_rejected(self) -> None:
        self.symlink(self.root / "dist", self.base, directory=True)
        with self.assertRaises(ValueError):
            exporter.export_source(self.root)
        self.assertFalse((self.base / "mikirin-source.zip").exists())

    def test_hardlinked_source_and_output_are_rejected(self) -> None:
        source = self.root / "README.md"
        alias = self.root / "export.zip"
        os.link(source, alias)
        with self.assertRaises(ValueError):
            exporter.export_source(self.root, alias)
        alias.unlink()
        secret = self.write(".env", "secret\n")
        leak = self.root / "tests/leak.py"
        leak.parent.mkdir()
        os.link(secret, leak)
        with self.assertRaises(ValueError):
            exporter.export_source(self.root)

    @unittest.skipUnless(os.name == "nt", "Windows junction test")
    def test_windows_junction_is_rejected(self) -> None:
        external = self.base / "outside"
        external.mkdir()
        junction = self.root / "docs"
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "New-Item -ItemType Junction -Path $env:EXPORT_TEST_LINK "
                 "-Target $env:EXPORT_TEST_TARGET -ErrorAction Stop | Out-Null"],
                env={**os.environ, "EXPORT_TEST_LINK": str(junction),
                     "EXPORT_TEST_TARGET": str(external)},
                check=True, capture_output=True,
            )
            with self.assertRaises(ValueError):
                exporter.export_source(self.root)
        finally:
            if junction.is_junction():
                junction.rmdir()

    def test_cli_works_from_another_directory_and_warns_about_git_history(self) -> None:
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--root", str(self.root),
             "--output", "release/source.zip"],
            cwd=self.base, capture_output=True, text=True, check=True,
        )
        self.assertTrue((self.root / "release/source.zip").is_file())
        self.assertIn("NEW repository", result.stdout)
        self.assertIn("Git history is unchanged", result.stdout)


if __name__ == "__main__":
    unittest.main()

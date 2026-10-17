"""Export an allowlisted working-tree snapshot, without Git or third-party tools.

Run: python scripts/export_source.py [--root PATH] [--output PATH.zip]
Relative output paths are relative to the project root, not the current directory.
Only the root files below and Python/Markdown resources in src/mikirin, Python
tests, Python scripts and scripts/README.md, docs/agent Markdown, and .github
YAML are eligible. Hidden, runtime, cache, scratch and common secret paths are
excluded at every depth. Links (including Windows reparse points and hardlinks)
in eligible inputs fail closed; output links are also rejected.

This is a path allowlist, NOT a content secret scanner: review allowed source,
Markdown, YAML and .env.example for embedded credentials before publishing.
It neither reads nor changes the Git index/history. Ignoring/exporting files
does not remove previously tracked artifacts or secrets from Git history. For
a clean GitHub publication, extract into a new directory and start a NEW repo.
Do not copy the old .git directory. Do not mutate the source tree during export.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import stat
import tempfile
import zipfile


ROOT_FILES = frozenset({
    "README.md", "pyproject.toml", "THIRD_PARTY_NOTICES.md", "LICENSE",
    ".env.example", ".gitignore", "run.ps1",
})
TREE_SUFFIXES = {
    "src/mikirin": {".py", ".md"},
    "tests": {".py"},
    "scripts": {".py"},
    "docs": {".md"},
    "agent": {".md"},
    ".github": {".yaml", ".yml"},
}
EXCLUDED_DIRS = frozenset({
    "data", "work", "db", "cache", "caches", "__pycache__", "build", "dist",
    "venv", "env", "node_modules", "scratch", "_scratch", "tmp", "temp",
})
PRIVATE_NAME = re.compile(
    r"(?:^|[_.-])(?:secrets?|credentials?|private|local|scratch)(?:$|[_.-])",
    re.IGNORECASE,
)
DEFAULT_OUTPUT = Path("dist/mikirin-source.zip")


def _excluded(name: str) -> bool:
    lower = name.casefold()
    return (
        lower.startswith(".")
        or lower in EXCLUDED_DIRS
        or lower.endswith((".egg-info", ".dist-info"))
        or PRIVATE_NAME.search(name) is not None
    )


def _check_path(path: Path) -> None:
    """Use lstat before resolve/open: resolve alone would hide linked inputs."""
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        reparse = getattr(info, "st_file_attributes", 0) & 0x400
        if stat.S_ISLNK(info.st_mode) or reparse:
            raise ValueError(f"Links/reparse points are not allowed: {part}")
        if part != path and not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Path ancestor is not a directory: {part}")


def _regular_file(path: Path) -> os.stat_result:
    _check_path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"Expected a regular, non-hardlinked file: {path}")
    return info


def _walk(directory: Path, suffixes: set[str]):
    _check_path(directory)
    if not directory.exists():
        return
    if not directory.is_dir():
        raise ValueError(f"Expected source directory: {directory}")
    for child in sorted(directory.iterdir()):
        if _excluded(child.name):
            continue
        _check_path(child)
        if child.is_dir():
            yield from _walk(child, suffixes)
        elif child.suffix in suffixes:
            _regular_file(child)
            yield child


def select_source_files(root: Path) -> list[Path]:
    """Return sorted absolute files, including eligible uncommitted source."""
    root = Path(os.path.abspath(root))
    _check_path(root)
    if not root.is_dir():
        raise ValueError(f"Project root is not a directory: {root}")
    files = []
    for name in sorted(ROOT_FILES | {"scripts/README.md"}):
        path = root / name
        _check_path(path)
        if path.exists():
            _regular_file(path)
            files.append(path)
    for directory, suffixes in TREE_SUFFIXES.items():
        files.extend(_walk(root / directory, suffixes))
    # Only versioned, purpose-written benchmark questions are exportable JSON;
    # never broaden this to runtime manifests or user-labelled review artifacts.
    files.extend(_walk(root / "src/mikirin/resources/benchmarks", {".json"}))
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _validate_output(root: Path, output: Path, files: list[Path]) -> None:
    _check_path(output)
    if any(part.casefold() == ".git" for part in output.parts):
        raise ValueError("Output must not be inside Git metadata")
    # Disallow entire source trees, including not-yet-existing source paths.
    # Casefold also makes this rule safe on case-insensitive filesystems.
    try:
        relative = output.relative_to(root).as_posix().casefold()
    except ValueError:
        relative = ""  # An absolute destination outside the project is supported.
    if relative in {name.casefold() for name in ROOT_FILES} or any(
        relative == directory or relative.startswith(directory + "/")
        for directory in ("src", "tests", "scripts", "docs", "agent", ".github")
    ):
        raise ValueError("Output must be outside the source files/directories")
    if output.suffix.casefold() != ".zip":
        raise ValueError("Output must have a .zip extension")
    if output.exists():
        _regular_file(output)
        if any(output.samefile(source) for source in files):
            raise ValueError("Output must not overwrite a source file")


def _read_source(path: Path) -> bytes:
    before = _regular_file(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"Source changed during export: {path}")
        data = stream.read()
    after = _regular_file(path)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError(f"Source changed during export: {path}")
    return data


def export_source(root: Path, output: Path | None = None) -> Path:
    """Atomically replace the destination; identical inputs yield identical ZIPs."""
    root = Path(os.path.abspath(root))
    output = Path(output) if output is not None else DEFAULT_OUTPUT
    output = Path(os.path.abspath(output if output.is_absolute() else root / output))
    files = select_source_files(root)
    _validate_output(root, output, files)
    if not files:
        raise ValueError("No allowlisted source files found")
    output.parent.mkdir(parents=True, exist_ok=True)
    _check_path(output.parent)
    fd, temporary = tempfile.mkstemp(prefix=".mikirin-source-", suffix=".tmp", dir=output.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for source in files:
                    name = source.relative_to(root).as_posix()
                    if "\\" in name or ":" in name:
                        raise ValueError(f"Non-portable archive path: {name}")
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.create_system = 3
                    info.external_attr = (stat.S_IFREG | 0o644) << 16
                    archive.writestr(
                        info, _read_source(source),
                        compress_type=zipfile.ZIP_DEFLATED, compresslevel=9,
                    )
        _validate_output(root, output, files)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", "-o", type=Path, help="ZIP path (relative to --root)")
    args = parser.parse_args(argv)
    try:
        output = export_source(args.root, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Exported source: {output}")
    print("Review allowed files for embedded secrets before publishing. Git history is unchanged;")
    print("extract into a NEW repository for clean publication; do not copy the old .git directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

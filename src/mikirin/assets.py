from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from mikirin.atomic import atomic_write_json
from mikirin.layout import ProjectLayout


_SOURCE_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RawAsset:
    id: str
    source: str
    source_version: str
    original_name: str
    relative_path: str
    content_hash: str
    byte_size: int
    ingested_at: str
    license_id: str
    redistributable: bool
    attribution_required: bool
    commercial_use_allowed: bool


class RawAssetStore:
    """Content-addressed, append-only storage for already acquired source files."""

    def __init__(self, layout: ProjectLayout) -> None:
        self.layout = layout

    def ingest(
        self,
        source_path: Path,
        *,
        source: str,
        source_version: str,
        acquired_on: date | None = None,
        license_id: str = "unknown",
        redistributable: bool = False,
        attribution_required: bool = False,
        commercial_use_allowed: bool = False,
    ) -> RawAsset:
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"待归档文件不存在: {source_path}")
        source = source.strip().lower()
        if not _SOURCE_RE.fullmatch(source):
            raise ValueError(f"source 必须匹配 {_SOURCE_RE.pattern!r}: {source!r}")
        source_version = source_version.strip()
        if not source_version:
            raise ValueError("source_version 不能为空")

        self.layout.initialize()
        digest = sha256_file(source_path)
        asset_id = f"{source}:sha256:{digest}"
        manifest_path = self.layout.manifests / "assets" / source / f"{digest}.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                return RawAsset(**json.load(handle))

        day = acquired_on or datetime.now(UTC).date()
        suffix = "".join(source_path.suffixes).lower()
        destination = self.layout.raw / source / day.isoformat() / f"{digest}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{digest}.", suffix=".tmp", dir=destination.parent
            )
            os.close(fd)
            temp_path = Path(temp_name)
            try:
                shutil.copyfile(source_path, temp_path)
                if sha256_file(temp_path) != digest:
                    raise RuntimeError("归档复制后的哈希不一致")
                os.replace(temp_path, destination)
            except BaseException:
                temp_path.unlink(missing_ok=True)
                raise

        record = RawAsset(
            id=asset_id,
            source=source,
            source_version=source_version,
            original_name=source_path.name,
            relative_path=destination.relative_to(self.layout.root).as_posix(),
            content_hash=f"sha256:{digest}",
            byte_size=source_path.stat().st_size,
            ingested_at=datetime.now(UTC).isoformat(),
            license_id=license_id,
            redistributable=redistributable,
            attribution_required=attribution_required,
            commercial_use_allowed=commercial_use_allowed,
        )
        atomic_write_json(manifest_path, asdict(record))
        return record


def load_raw_asset_manifest(path: Path) -> RawAsset:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return RawAsset(**payload)

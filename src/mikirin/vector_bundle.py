"""Side-build and atomically publish local vector-index bundles."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mikirin.atomic import atomic_write_json
from mikirin.vector_index import VectorIndex, _stable_hash


BUNDLE_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class VectorBundle:
    directory: Path
    bundle_fingerprint: str
    embedding_fingerprint: str
    indexes: dict[str, dict[str, object]]
    source_database: dict[str, object]


class VectorBundlePublisher:
    """Keep immutable version directories and atomically switch current.json."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.versions = self.root / "versions"
        self.pointer = self.root / "current.json"

    def create_staging(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        directory = self.root / f".building-{uuid.uuid4().hex}"
        directory.mkdir()
        return directory

    def write_manifest(
        self,
        directory: Path,
        *,
        embedding_fingerprint: str,
        index_directories: dict[str, Path],
        source_database: dict[str, object],
    ) -> VectorBundle:
        directory = directory.resolve()
        self._require_child(directory)
        indexes: dict[str, dict[str, object]] = {}
        for name, index_directory in sorted(index_directories.items()):
            index_directory = index_directory.resolve()
            self._require_child(index_directory, parent=directory)
            index = VectorIndex(index_directory)
            try:
                if index.embedding_fingerprint != embedding_fingerprint:
                    raise RuntimeError(
                        f"embedding fingerprint mismatch in index {name}: "
                        f"{index.embedding_fingerprint} != {embedding_fingerprint}"
                    )
                indexes[name] = {
                    "path": index_directory.relative_to(directory).as_posix(),
                    "count": index.count,
                    "dimensions": index.dimensions,
                    "index_fingerprint": index.index_fingerprint,
                    "embedding_fingerprint": index.embedding_fingerprint,
                }
            finally:
                index.close()
        if not indexes:
            raise RuntimeError("refusing to publish a vector bundle without indexes")
        bundle_contract = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "embedding_fingerprint": embedding_fingerprint,
            "indexes": {
                name: {
                    "count": value["count"],
                    "index_fingerprint": value["index_fingerprint"],
                }
                for name, value in indexes.items()
            },
            "source_database": source_database,
        }
        fingerprint = _stable_hash(bundle_contract)
        payload = {
            **bundle_contract,
            "bundle_fingerprint": fingerprint,
            "created_at": _utc_now(),
            "indexes": indexes,
        }
        atomic_write_json(directory / "manifest.json", payload)
        return VectorBundle(
            directory=directory,
            bundle_fingerprint=fingerprint,
            embedding_fingerprint=embedding_fingerprint,
            indexes=indexes,
            source_database=source_database,
        )

    def validate(
        self, directory: Path, *, expected_embedding_fingerprint: str | None = None
    ) -> VectorBundle:
        directory = directory.resolve()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", 0)) != BUNDLE_SCHEMA_VERSION:
            raise RuntimeError("unsupported vector bundle schema version")
        embedding_fingerprint = str(manifest["embedding_fingerprint"])
        if (
            expected_embedding_fingerprint is not None
            and embedding_fingerprint != expected_embedding_fingerprint
        ):
            raise RuntimeError("runtime embedding fingerprint does not match vector bundle")
        indexes = manifest.get("indexes")
        if not isinstance(indexes, dict) or not indexes:
            raise RuntimeError("vector bundle has no indexes")
        for name, raw in indexes.items():
            if not isinstance(raw, dict):
                raise RuntimeError(f"invalid vector index manifest entry: {name}")
            index_directory = (directory / str(raw["path"])).resolve()
            self._require_child(index_directory, parent=directory)
            index = VectorIndex(index_directory)
            try:
                if index.embedding_fingerprint != embedding_fingerprint:
                    raise RuntimeError(f"embedding fingerprint mismatch in {name}")
                if index.index_fingerprint != raw.get("index_fingerprint"):
                    raise RuntimeError(f"index fingerprint mismatch in {name}")
                if index.count != int(raw.get("count", -1)):
                    raise RuntimeError(f"index count mismatch in {name}")
            finally:
                index.close()
        contract = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "embedding_fingerprint": embedding_fingerprint,
            "indexes": {
                name: {
                    "count": raw["count"],
                    "index_fingerprint": raw["index_fingerprint"],
                }
                for name, raw in indexes.items()
            },
            "source_database": manifest.get("source_database", {}),
        }
        fingerprint = _stable_hash(contract)
        if fingerprint != manifest.get("bundle_fingerprint"):
            raise RuntimeError("vector bundle fingerprint mismatch")
        return VectorBundle(
            directory=directory,
            bundle_fingerprint=fingerprint,
            embedding_fingerprint=embedding_fingerprint,
            indexes=indexes,
            source_database=manifest.get("source_database", {}),
        )

    def publish(self, staging: Path) -> VectorBundle:
        staging = staging.resolve()
        self._require_child(staging)
        bundle = self.validate(staging)
        self.versions.mkdir(parents=True, exist_ok=True)
        target = self.versions / bundle.bundle_fingerprint
        if target.exists():
            raise FileExistsError(f"vector bundle version already exists: {target}")
        os.replace(staging, target)
        atomic_write_json(
            self.pointer,
            {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "bundle_fingerprint": bundle.bundle_fingerprint,
                "embedding_fingerprint": bundle.embedding_fingerprint,
                "directory": target.relative_to(self.root).as_posix(),
                "published_at": _utc_now(),
            },
        )
        return self.validate(target)

    def current(
        self, *, expected_embedding_fingerprint: str | None = None
    ) -> VectorBundle | None:
        if not self.pointer.exists():
            return None
        pointer = json.loads(self.pointer.read_text(encoding="utf-8"))
        directory = (self.root / str(pointer["directory"])).resolve()
        self._require_child(directory)
        bundle = self.validate(
            directory, expected_embedding_fingerprint=expected_embedding_fingerprint
        )
        if bundle.bundle_fingerprint != pointer.get("bundle_fingerprint"):
            raise RuntimeError("vector current pointer fingerprint mismatch")
        return bundle

    def _require_child(self, path: Path, *, parent: Path | None = None) -> None:
        base = (parent or self.root).resolve()
        try:
            path.resolve().relative_to(base)
        except ValueError as error:
            raise RuntimeError(f"vector bundle path escapes root: {path}") from error


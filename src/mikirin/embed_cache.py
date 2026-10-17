"""Scalable content-addressed embedding cache.

Pre-v2 one-vector-per-file entries remain readable for zero-cost migration. All
new writes use batched SQLite transactions so million-row builds do not create
a million filesystem entries.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import threading
from pathlib import Path
from typing import Iterable


class EmbeddingCache:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.database = directory / "cache-v2.sqlite"
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.database, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS embedding_cache("
            "cache_key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, "
            "dimensions INTEGER NOT NULL, vector_int8 BLOB NOT NULL, "
            "quantized_norm REAL NOT NULL DEFAULT 0) WITHOUT ROWID"
        )
        columns = {
            str(row[1]) for row in self._db.execute("PRAGMA table_info(embedding_cache)")
        }
        if "quantized_norm" not in columns:
            self._db.execute(
                "ALTER TABLE embedding_cache ADD COLUMN "
                "quantized_norm REAL NOT NULL DEFAULT 0"
            )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_embedding_cache_fingerprint "
            "ON embedding_cache(fingerprint)"
        )
        self._db.commit()
        self._closed = False

    @staticmethod
    def _key(fingerprint: str, text: str) -> str:
        return hashlib.sha256((fingerprint + "\n" + text).encode("utf-8")).hexdigest()

    def _legacy_path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.int8"

    @staticmethod
    def _decode(raw: bytes, dimensions: int | None) -> list[float] | None:
        if not raw or (dimensions is not None and len(raw) != dimensions):
            return None
        values = [(value - 128) / 127 for value in raw]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    @staticmethod
    def _encode(vector: list[float]) -> bytes:
        if not vector or not all(math.isfinite(float(value)) for value in vector):
            raise ValueError("embedding cache only accepts finite, non-empty vectors")
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return bytes(
            max(0, min(255, round(value / norm * 127 + 128))) for value in vector
        )

    @staticmethod
    def _quantized_norm(raw: bytes) -> float:
        return math.sqrt(sum((value - 128) * (value - 128) for value in raw)) or 1.0

    def get(
        self, fingerprint: str, text: str, *, dimensions: int | None = None
    ) -> list[float] | None:
        raw = self.get_int8(fingerprint, text, dimensions=dimensions)
        return None if raw is None else self._decode(raw, dimensions)

    def get_int8(
        self, fingerprint: str, text: str, *, dimensions: int | None = None
    ) -> bytes | None:
        result = self.get_quantized(fingerprint, text, dimensions=dimensions)
        return None if result is None else result[0]

    def get_quantized(
        self, fingerprint: str, text: str, *, dimensions: int | None = None
    ) -> tuple[bytes, float] | None:
        if self._closed:
            raise RuntimeError("embedding cache is closed")
        key = self._key(fingerprint, text)
        with self._lock:
            row = self._db.execute(
                "SELECT dimensions, vector_int8, quantized_norm "
                "FROM embedding_cache WHERE cache_key=?",
                (key,),
            ).fetchone()
        if row is not None:
            if dimensions is not None and int(row[0]) != dimensions:
                return None
            raw = bytes(row[1])
            if dimensions is not None and len(raw) != dimensions:
                return None
            norm = float(row[2])
            return raw, norm if norm > 0 else self._quantized_norm(raw)
        legacy = self._legacy_path(key)
        if not legacy.exists():
            return None
        raw = legacy.read_bytes()
        if not raw or (dimensions is not None and len(raw) != dimensions):
            return None
        return raw, self._quantized_norm(raw)

    def put(self, fingerprint: str, text: str, vector: list[float]) -> None:
        self.put_many(fingerprint, [(text, vector)])

    def put_many(
        self, fingerprint: str, items: Iterable[tuple[str, list[float]]]
    ) -> int:
        if self._closed:
            raise RuntimeError("embedding cache is closed")
        rows = []
        for text, vector in items:
            raw = self._encode(vector)
            rows.append(
                (
                    self._key(fingerprint, text),
                    fingerprint,
                    len(raw),
                    raw,
                    self._quantized_norm(raw),
                )
            )
        if not rows:
            return 0
        with self._lock, self._db:
            self._db.executemany(
                "INSERT OR IGNORE INTO embedding_cache("
                "cache_key, fingerprint, dimensions, vector_int8, quantized_norm) "
                "VALUES(?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def put_int8_many(
        self, fingerprint: str, items: Iterable[tuple[str, bytes]]
    ) -> int:
        if self._closed:
            raise RuntimeError("embedding cache is closed")
        return self.put_quantized_many(
            fingerprint,
            (
                (text, raw, self._quantized_norm(raw))
                for text, raw in items
            ),
        )

    def put_quantized_many(
        self, fingerprint: str, items: Iterable[tuple[str, bytes, float]]
    ) -> int:
        rows = []
        for text, raw, norm in items:
            if not raw:
                raise ValueError("embedding cache only accepts non-empty int8 vectors")
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("quantized vector norm must be finite and positive")
            rows.append((self._key(fingerprint, text), fingerprint, len(raw), raw, norm))
        if not rows:
            return 0
        with self._lock, self._db:
            self._db.executemany(
                "INSERT INTO embedding_cache("
                "cache_key, fingerprint, dimensions, vector_int8, quantized_norm) "
                "VALUES(?,?,?,?,?) ON CONFLICT(cache_key) DO UPDATE SET "
                "quantized_norm=CASE WHEN embedding_cache.quantized_norm<=0 "
                "THEN excluded.quantized_norm ELSE embedding_cache.quantized_norm END",
                rows,
            )
        return len(rows)

    def promote(
        self,
        *,
        source_fingerprint: str,
        source_text: str,
        target_fingerprint: str,
        target_text: str,
        dimensions: int,
    ) -> bool:
        vector = self.get(source_fingerprint, source_text, dimensions=dimensions)
        if vector is None:
            return False
        self.put(target_fingerprint, target_text, vector)
        return True

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._db.close()
            self._closed = True

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

"""embedding 客户端与本地两级向量索引（运行时纯标准库）。

- EmbeddingClient：OpenAI 兼容 /v1/embeddings，批量+重试；
  指纹绑定（provider/base_url/model/dimensions/prefix/normalize）。
- int8 量化 + 随机投影 SimHash 签名，固定宽度二进制文件 + mmap 粗排，
  纯 Python 点积精排。离线侧仅用标准库（投影矩阵用 random 模块生成，
  不依赖 NumPy——matrix 存进索引头保证可复现）。
"""

from __future__ import annotations

import json
import hashlib
import heapq
import math
import mmap
import random
import struct
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit

from mikirin.atomic import atomic_write_json


INPUT_NORMALIZATION_VERSION = "unicode-nfc-lf-v1"
QUANTIZATION_KIND = "symmetric-int8"
QUANTIZATION_VERSION = "v1"
INDEX_FORMAT_VERSION = 2
METADATA_LAYOUT_VERSION = "vector-meta-v2-scope-indexes"


class PermanentEmbeddingError(RuntimeError):
    """A request that needs a configuration or billing change before retrying."""


def canonical_base_url(value: str) -> str:
    """Canonicalize a provider endpoint without changing its path semantics."""
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError("embedding base_url must be an absolute HTTPS URL")
    path = parsed.path.rstrip("/")
    return urlunsplit(("https", parsed.netloc.lower(), path, "", ""))


def normalize_embedding_input(
    text: str, version: str = INPUT_NORMALIZATION_VERSION
) -> str:
    if version != INPUT_NORMALIZATION_VERSION:
        raise ValueError(f"unsupported input normalization version: {version}")
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    provider: str
    base_url: str
    model: str
    dimensions: int
    distance: str = "cosine"
    query_prefix: str = ""
    document_prefix: str = ""
    normalize: bool = True
    input_normalization_version: str = INPUT_NORMALIZATION_VERSION

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip():
            raise ValueError("embedding provider and model cannot be empty")
        canonical_base_url(self.base_url)
        if self.dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        if self.distance != "cosine":
            raise ValueError("only cosine embedding distance is supported")
        if not isinstance(self.normalize, bool):
            raise ValueError("embedding normalize must be boolean")
        normalize_embedding_input("", self.input_normalization_version)

    def contract(self) -> dict[str, object]:
        return {
            "provider": self.provider.strip(),
            "canonical_base_url": canonical_base_url(self.base_url),
            "model": self.model.strip(),
            "returned_dimensions": self.dimensions,
            "normalize": self.normalize,
            "query_prefix": self.query_prefix,
            "document_prefix": self.document_prefix,
            "input_normalization_version": self.input_normalization_version,
        }

    def fingerprint(self) -> str:
        return _stable_hash(self.contract())

    def legacy_fingerprint(self) -> str:
        """Pre-v2 fingerprint, only for one-time cache/index migration."""
        return "|".join(
            [
                self.provider,
                self.base_url.rstrip("/"),
                self.model,
                str(self.dimensions),
                str(self.normalize),
                self.query_prefix,
                self.document_prefix,
            ]
        )


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig, api_key: str,
                 batch_size: int = 32, max_retries: int = 3) -> None:
        self.config = config
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_retries = max_retries

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(
            [
                self.config.document_prefix
                + normalize_embedding_input(t, self.config.input_normalization_version)
                for t in texts
            ]
        )

    def embed_query(self, text: str) -> list[float]:
        return self._embed(
            [
                self.config.query_prefix
                + normalize_embedding_input(text, self.config.input_normalization_version)
            ]
        )[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            payload = json.dumps(
                {"model": self.config.model, "input": batch}
            ).encode("utf-8")
            last_error: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    request = urllib.request.Request(
                        self.config.base_url.rstrip("/") + "/embeddings",
                        data=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                    )
                    with urllib.request.urlopen(request, timeout=60) as response:
                        body = json.loads(response.read())
                    returned = [
                        item["embedding"]
                        for item in sorted(body["data"], key=lambda item: item["index"])
                    ]
                    if len(returned) != len(batch):
                        raise RuntimeError(
                            f"embedding response count mismatch: {len(returned)} != {len(batch)}"
                        )
                    for vector in returned:
                        if len(vector) != self.config.dimensions:
                            raise RuntimeError(
                                "embedding returned dimensions mismatch: "
                                f"{len(vector)} != {self.config.dimensions}"
                            )
                        if not all(math.isfinite(float(value)) for value in vector):
                            raise RuntimeError("embedding response contains non-finite values")
                    vectors.extend(returned)
                    break
                except urllib.error.HTTPError as error:
                    if error.code in {400, 401, 402, 403, 404}:
                        raise PermanentEmbeddingError(
                            f"embedding request permanently rejected: HTTP {error.code}"
                        ) from error
                    last_error = error
                    time.sleep(2**attempt)
                except Exception as error:  # 网络/限流/服务错误统一重试
                    last_error = error
                    time.sleep(2**attempt)
            else:
                raise RuntimeError(f"embedding request failed: {last_error}")
        return vectors


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [x / norm for x in vector]


def quantize_int8(vector: list[float]) -> bytes:
    """归一化后按 [-1,1]→[-127,127] 量化。"""
    unit = normalize_vector(vector)
    return bytes(max(0, min(255, round(x * 127 + 128))) for x in unit)


def _int8_dot(a: bytes, b: bytes) -> int:
    return sum((x - 128) * (y - 128) for x, y in zip(a, b))


def build_projection(dimensions: int, bits: int, seed: int) -> list[int]:
    """生成随机投影（每 bit 一个随机维度+符号），存入索引头可复现。"""
    rng = random.Random(seed)
    return [rng.randrange(dimensions) * 2 + rng.choice((0, 1)) for _ in range(bits)]


def projection_hash(projection: list[int]) -> str:
    return _stable_hash(projection)


def make_index_fingerprint(
    *,
    embedding_fingerprint: str,
    projection: list[int],
    signature_bits: int,
    schema_versions: dict[str, str],
    coarse_retrieval: dict[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    contract: dict[str, object] = {
        "embedding_fingerprint": embedding_fingerprint,
        "projection_hash": projection_hash(projection),
        "signature_bits": signature_bits,
        "quantization_kind": QUANTIZATION_KIND,
        "quantization_version": QUANTIZATION_VERSION,
        "metadata_layout_version": METADATA_LAYOUT_VERSION,
        "schema_versions": dict(sorted(schema_versions.items())),
        "coarse_retrieval": coarse_retrieval or {"kind": "full-signature-scan"},
    }
    return _stable_hash(contract), contract


def simhash_sign(int8_vector: bytes, projection: list[int]) -> int:
    signature = 0
    for bit, packed in enumerate(projection):
        dim, sign = packed // 2, packed % 2
        value = int8_vector[dim] - 128
        if (value > 0) == (sign == 1):
            signature |= 1 << bit
    return signature


def lsh_probe_keys(key: int, bits: int, radius: int) -> Iterator[int]:
    """Yield a deterministic Hamming multiprobe neighborhood up to radius two."""
    yield key
    if radius < 1:
        return
    for first in range(bits):
        yield key ^ (1 << first)
    if radius < 2:
        return
    for first in range(bits):
        for second in range(first + 1, bits):
            yield key ^ (1 << first) ^ (1 << second)


class VectorIndexWriter:
    """追加写固定宽度签名文件 + int8 向量文件 + SQLite 元数据。"""

    FILE_NAMES = ("header.json", "signatures.bin", "vectors.int8", "norms.bin", "meta.db")

    def __init__(
        self,
        directory: Path,
        *,
        dimensions: int,
        bits: int,
        seed: int = 20260729,
        embedding_fingerprint: str = "",
        schema_versions: dict[str, str] | None = None,
        lsh_bands: int = 0,
        lsh_band_bits: int = 16,
        lsh_probe_radius: int = 2,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        if bits < 8 or bits % 8:
            raise ValueError("signature bits must be a positive multiple of 8")
        if lsh_bands < 0 or lsh_band_bits < 1 or lsh_band_bits % 8:
            raise ValueError("LSH bands must be non-negative and band bits a multiple of 8")
        if lsh_bands * lsh_band_bits > bits:
            raise ValueError("LSH bands exceed the signature width")
        if lsh_probe_radius < 0 or lsh_probe_radius > 2:
            raise ValueError("LSH probe radius must be between 0 and 2")
        directory.mkdir(parents=True, exist_ok=True)
        existing = [name for name in self.FILE_NAMES if (directory / name).exists()]
        if existing:
            raise FileExistsError(
                f"vector index directory is not empty; use a side build: {existing}"
            )
        self.directory = directory
        self.dimensions = dimensions
        self.bits = bits
        self.seed = seed
        self.projection = build_projection(dimensions, bits, seed)
        self.embedding_fingerprint = embedding_fingerprint
        self.schema_versions = schema_versions or {}
        self.lsh_bands = lsh_bands
        self.lsh_band_bits = lsh_band_bits
        self.lsh_probe_radius = lsh_probe_radius
        coarse_retrieval = (
            {
                "kind": "binary-band-lsh",
                "bands": lsh_bands,
                "band_bits": lsh_band_bits,
                "probe_radius": lsh_probe_radius,
                "format_version": 1,
            }
            if lsh_bands
            else {"kind": "full-signature-scan"}
        )
        self.index_fingerprint, self.index_contract = make_index_fingerprint(
            embedding_fingerprint=embedding_fingerprint,
            projection=self.projection,
            signature_bits=bits,
            schema_versions=self.schema_versions,
            coarse_retrieval=coarse_retrieval,
        )
        self._sig = (directory / "signatures.bin").open("wb")
        self._vec = (directory / "vectors.int8").open("wb")
        self._norm = (directory / "norms.bin").open("wb")
        self._sig_digest = hashlib.sha256()
        self._vec_digest = hashlib.sha256()
        self._norm_digest = hashlib.sha256()
        self._row = 0
        self._closed = False
        import sqlite3

        self._db = sqlite3.connect(directory / "meta.db")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS vector_meta("
            "row_id INTEGER PRIMARY KEY, object_kind TEXT NOT NULL, "
            "object_id TEXT NOT NULL, subject_entity_id TEXT, episode_entity_id TEXT, "
            "document_id TEXT, metadata_json TEXT NOT NULL DEFAULT '{}', "
            "UNIQUE(object_kind, object_id))"
        )
        atomic_write_json(
            directory / "header.json",
            {
                "format_version": INDEX_FORMAT_VERSION,
                "complete": False,
                "dimensions": dimensions,
                "bits": bits,
                "seed": seed,
                "projection": self.projection,
                "embedding_fingerprint": embedding_fingerprint,
                "index_fingerprint": self.index_fingerprint,
                "index_contract": self.index_contract,
            },
        )

    def add(
        self,
        object_kind: str,
        object_id: str,
        vector: list[float],
        *,
        subject_entity_id: str | None = None,
        episode_entity_id: str | None = None,
        document_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"vector dimensions mismatch: {len(vector)} != {self.dimensions}"
            )
        int8 = quantize_int8(vector)
        self.add_int8(
            object_kind,
            object_id,
            int8,
            subject_entity_id=subject_entity_id,
            episode_entity_id=episode_entity_id,
            document_id=document_id,
            metadata=metadata,
        )

    def add_int8(
        self,
        object_kind: str,
        object_id: str,
        int8: bytes,
        *,
        subject_entity_id: str | None = None,
        episode_entity_id: str | None = None,
        document_id: str | None = None,
        metadata: dict[str, object] | None = None,
        norm: float | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("vector index writer is closed")
        if len(int8) != self.dimensions:
            raise ValueError(
                f"int8 vector dimensions mismatch: {len(int8)} != {self.dimensions}"
            )
        if not object_kind.strip() or not object_id.strip():
            raise ValueError("object_kind and object_id cannot be empty")
        signature = simhash_sign(int8, self.projection)
        signature_bytes = signature.to_bytes(self.bits // 8, "big")
        self._sig.write(signature_bytes)
        self._vec.write(int8)
        norm = norm or math.sqrt(sum((b - 128) * (b - 128) for b in int8)) or 1.0
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("quantized vector norm must be finite and positive")
        norm_bytes = struct.pack("f", norm)
        self._norm.write(norm_bytes)
        self._sig_digest.update(signature_bytes)
        self._vec_digest.update(int8)
        self._norm_digest.update(norm_bytes)
        self._db.execute(
            "INSERT INTO vector_meta("
            "row_id, object_kind, object_id, subject_entity_id, episode_entity_id, "
            "document_id, metadata_json) VALUES(?,?,?,?,?,?,?)",
            (
                self._row,
                object_kind,
                object_id,
                subject_entity_id,
                episode_entity_id,
                document_id,
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
            ),
        )
        self._row += 1

    def close(self) -> None:
        if self._closed:
            return
        if self._row == 0:
            raise RuntimeError("refusing to finalize an empty vector index")
        self._sig.close()
        self._vec.close()
        self._norm.close()
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS vector_meta_subject_row "
            "ON vector_meta(subject_entity_id, row_id)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS vector_meta_episode_row "
            "ON vector_meta(episode_entity_id, row_id)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS vector_meta_kind_row "
            "ON vector_meta(object_kind, row_id)"
        )
        self._db.commit()
        self._db.close()
        self._closed = True
        lsh_files = self._build_lsh() if self.lsh_bands else {}
        atomic_write_json(
            self.directory / "header.json",
            {
                "format_version": INDEX_FORMAT_VERSION,
                "complete": True,
                "count": self._row,
                "dimensions": self.dimensions,
                "bits": self.bits,
                "seed": self.seed,
                "projection": self.projection,
                "embedding_fingerprint": self.embedding_fingerprint,
                "index_fingerprint": self.index_fingerprint,
                "index_contract": self.index_contract,
                "lsh": {
                    "bands": self.lsh_bands,
                    "band_bits": self.lsh_band_bits,
                    "probe_radius": self.lsh_probe_radius,
                    "files": lsh_files,
                }
                if self.lsh_bands
                else None,
                "files": {
                    "signatures.bin": self._sig_digest.hexdigest(),
                    "vectors.int8": self._vec_digest.hexdigest(),
                    "norms.bin": self._norm_digest.hexdigest(),
                },
            },
        )

    def _build_lsh(self) -> dict[str, dict[str, object]]:
        """Build compact per-band offset/posting files without SQLite lookup."""
        signature_path = self.directory / "signatures.bin"
        step = self.bits // 8
        width = self.lsh_band_bits // 8
        key_count = 1 << self.lsh_band_bits
        files: dict[str, dict[str, object]] = {}
        for band in range(self.lsh_bands):
            buckets: dict[int, array[int]] = {}
            offset = band * width
            with signature_path.open("rb") as handle:
                for row_id in range(self._row):
                    signature = handle.read(step)
                    if len(signature) != step:
                        raise RuntimeError("signature file truncated during LSH build")
                    key = int.from_bytes(signature[offset : offset + width], "big")
                    buckets.setdefault(key, array("I")).append(row_id)
            offsets = array("Q", [0])
            postings = array("I")
            for key in range(key_count):
                rows = buckets.get(key)
                if rows:
                    postings.extend(rows)
                offsets.append(len(postings))
            if sys.byteorder != "little":
                offsets.byteswap()
                postings.byteswap()
            offsets_name = f"lsh-band-{band:02d}.offsets"
            rows_name = f"lsh-band-{band:02d}.rows"
            offsets_bytes = offsets.tobytes()
            rows_bytes = postings.tobytes()
            from mikirin.atomic import atomic_write_bytes

            atomic_write_bytes(self.directory / offsets_name, offsets_bytes)
            atomic_write_bytes(self.directory / rows_name, rows_bytes)
            files[str(band)] = {
                "offsets": offsets_name,
                "rows": rows_name,
                "postings": len(postings),
                "offsets_sha256": hashlib.sha256(offsets_bytes).hexdigest(),
                "rows_sha256": hashlib.sha256(rows_bytes).hexdigest(),
            }
        return files

    def __enter__(self) -> "VectorIndexWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
            return
        if not self._closed:
            self._sig.close()
            self._vec.close()
            self._norm.close()
            self._db.close()
            self._closed = True


class VectorIndex:
    """只读索引：mmap 签名粗排 + int8 精排。"""

    def __init__(self, directory: Path) -> None:
        header = json.loads((directory / "header.json").read_text(encoding="utf-8"))
        self.format_version = int(header.get("format_version", 1))
        if self.format_version >= INDEX_FORMAT_VERSION and not header.get("complete", False):
            raise RuntimeError(f"vector index is incomplete: {directory}")
        self.dimensions = int(header["dimensions"])
        self.bits = int(header["bits"])
        if self.dimensions < 1 or self.bits < 8 or self.bits % 8:
            raise RuntimeError("invalid vector index dimensions or signature width")
        self.projection = [int(value) for value in header["projection"]]
        if len(self.projection) != self.bits:
            raise RuntimeError("vector index projection length does not match signature bits")
        self.embedding_fingerprint = str(header.get("embedding_fingerprint", ""))
        self.index_fingerprint = str(header.get("index_fingerprint", ""))
        self.index_contract = header.get("index_contract", {})
        self._sig_bytes = header["bits"] // 8
        self._sig_file = (directory / "signatures.bin").open("rb")
        self._sig_mm = mmap.mmap(self._sig_file.fileno(), 0, access=mmap.ACCESS_READ)
        self._vec_file = (directory / "vectors.int8").open("rb")
        self._vec_mm = mmap.mmap(self._vec_file.fileno(), 0, access=mmap.ACCESS_READ)
        self._norm_file = (directory / "norms.bin").open("rb")
        self._norm_mm = mmap.mmap(self._norm_file.fileno(), 0, access=mmap.ACCESS_READ)
        sizes = {
            "signatures": len(self._sig_mm),
            "vectors": len(self._vec_mm),
            "norms": len(self._norm_mm),
        }
        if sizes["signatures"] % self._sig_bytes:
            raise RuntimeError("truncated vector signature file")
        self.count = sizes["signatures"] // self._sig_bytes
        if sizes["vectors"] != self.count * self.dimensions:
            raise RuntimeError("vector file size does not match index count")
        if sizes["norms"] != self.count * 4:
            raise RuntimeError("norm file size does not match index count")
        if int(header.get("count", self.count)) != self.count:
            raise RuntimeError("header count does not match vector files")
        import sqlite3

        self._db = sqlite3.connect(
            (directory / "meta.db").resolve().as_uri() + "?mode=ro",
            uri=True,
            check_same_thread=False,  # 只读连接，允许 to_thread 工作线程使用
        )
        meta_count = int(self._db.execute("SELECT count(*) FROM vector_meta").fetchone()[0])
        if meta_count != self.count:
            raise RuntimeError(
                f"vector metadata count mismatch: {meta_count} != {self.count}"
            )
        self._lsh: list[tuple[object, mmap.mmap, object, mmap.mmap]] = []
        lsh = header.get("lsh")
        self.lsh_bands = 0
        self.lsh_band_bits = 0
        self.lsh_probe_radius = 0
        if isinstance(lsh, dict) and int(lsh.get("bands", 0)):
            self.lsh_bands = int(lsh["bands"])
            self.lsh_band_bits = int(lsh["band_bits"])
            self.lsh_probe_radius = int(lsh["probe_radius"])
            files = lsh.get("files", {})
            for band in range(self.lsh_bands):
                raw = files.get(str(band), {})
                offsets_file = (directory / str(raw["offsets"])).open("rb")
                offsets_mm = mmap.mmap(
                    offsets_file.fileno(), 0, access=mmap.ACCESS_READ
                )
                rows_file = (directory / str(raw["rows"])).open("rb")
                rows_mm = mmap.mmap(rows_file.fileno(), 0, access=mmap.ACCESS_READ)
                expected_offsets = ((1 << self.lsh_band_bits) + 1) * 8
                if len(offsets_mm) != expected_offsets or len(rows_mm) != self.count * 4:
                    offsets_mm.close()
                    rows_mm.close()
                    offsets_file.close()
                    rows_file.close()
                    raise RuntimeError("invalid LSH offset or posting file size")
                self._lsh.append((offsets_file, offsets_mm, rows_file, rows_mm))

    def _rows_for_scope(
        self,
        *,
        object_kinds: set[str] | None,
        subject_entity_id: str | None,
        episode_entity_id: str | None,
    ) -> range | list[int]:
        if not object_kinds and subject_entity_id is None and episode_entity_id is None:
            return range(self.count)
        clauses: list[str] = []
        parameters: list[object] = []
        if object_kinds:
            marks = ",".join("?" for _ in object_kinds)
            clauses.append(f"object_kind IN ({marks})")
            parameters.extend(sorted(object_kinds))
        if subject_entity_id is not None:
            clauses.append("subject_entity_id=?")
            parameters.append(subject_entity_id)
        if episode_entity_id is not None:
            clauses.append("episode_entity_id=?")
            parameters.append(episode_entity_id)
        where = " AND ".join(clauses) or "1"
        return [
            int(row[0])
            for row in self._db.execute(
                f"SELECT row_id FROM vector_meta WHERE {where} ORDER BY row_id",
                parameters,
            )
        ]

    def search(
        self,
        query_vector: list[float],
        *,
        coarse: int = 500,
        top: int = 50,
        object_kinds: set[str] | None = None,
        subject_entity_id: str | None = None,
        episode_entity_id: str | None = None,
    ) -> list[tuple[str, str, float]]:
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"query vector dimensions mismatch: {len(query_vector)} != {self.dimensions}"
            )
        if coarse < 1 or top < 1:
            return []
        int8 = quantize_int8(query_vector)
        signature = simhash_sign(int8, self.projection)
        step = self._sig_bytes
        rows = self._rows_for_scope(
            object_kinds=object_kinds,
            subject_entity_id=subject_entity_id,
            episode_entity_id=episode_entity_id,
        )
        if self._lsh:
            lsh_rows = self._lsh_candidates(signature)
            if isinstance(rows, range):
                rows = lsh_rows
            elif len(rows) > 50_000:
                allowed = set(rows)
                rows = [row for row in lsh_rows if row in allowed]
        candidates = heapq.nsmallest(
            coarse,
            (
                (
                    (
                        signature
                        ^ int.from_bytes(
                            self._sig_mm[row * step : row * step + step], "big"
                        )
                    ).bit_count(),
                    row,
                )
                for row in rows
            ),
        )
        query_norm = math.sqrt(sum((b - 128) * (b - 128) for b in int8)) or 1.0
        refined: list[tuple[float, int]] = []
        for _distance, row in candidates:
            start = row * self.dimensions
            vector = self._vec_mm[start : start + self.dimensions]
            (stored_norm,) = struct.unpack("f", self._norm_mm[row * 4 : row * 4 + 4])
            cosine = _int8_dot(int8, vector) / (query_norm * stored_norm)
            refined.append((cosine, row))
        refined.sort(key=lambda item: -item[0])
        results = []
        for score, row in refined[:top]:
            meta = self._db.execute(
                "SELECT object_kind, object_id FROM vector_meta WHERE row_id=?",
                (row,),
            ).fetchone()
            if meta:
                results.append((meta[0], meta[1], score))
        return results

    def _lsh_candidates(self, signature: int) -> list[int]:
        signature_bytes = signature.to_bytes(self._sig_bytes, "big")
        width = self.lsh_band_bits // 8
        candidates: set[int] = set()
        for band, (_offset_file, offsets, _rows_file, rows) in enumerate(self._lsh):
            start = band * width
            key = int.from_bytes(signature_bytes[start : start + width], "big")
            for probe in lsh_probe_keys(
                key, self.lsh_band_bits, self.lsh_probe_radius
            ):
                begin = struct.unpack_from("<Q", offsets, probe * 8)[0]
                end = struct.unpack_from("<Q", offsets, (probe + 1) * 8)[0]
                for position in range(begin, end):
                    candidates.add(struct.unpack_from("<I", rows, position * 4)[0])
        return sorted(candidates)

    def metadata(self, row_id: int) -> dict[str, object] | None:
        """Return normalized metadata for diagnostics and late aggregation."""
        row = self._db.execute(
            "SELECT object_kind, object_id, subject_entity_id, episode_entity_id, "
            "document_id, metadata_json FROM vector_meta WHERE row_id=?",
            (row_id,),
        ).fetchone()
        if row is None:
            return None
        try:
            extra = json.loads(row[5])
        except (TypeError, ValueError, json.JSONDecodeError):
            extra = {}
        return {
            "row_id": row_id,
            "object_kind": row[0],
            "object_id": row[1],
            "subject_entity_id": row[2],
            "episode_entity_id": row[3],
            "document_id": row[4],
            "metadata": extra,
        }

    def close(self) -> None:
        for offsets_file, offsets_mm, rows_file, rows_mm in self._lsh:
            offsets_mm.close()
            rows_mm.close()
            offsets_file.close()
            rows_file.close()
        self._sig_mm.close()
        self._vec_mm.close()
        self._norm_mm.close()
        self._sig_file.close()
        self._vec_file.close()
        self._norm_file.close()
        self._db.close()

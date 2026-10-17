"""语义单集检索：无锚查询的向量兜底（A3.2 运行时侧）。

路由位置：明确实体/集数 → 对话焦点 → 别名与 FTS → 【这里】→ LLM 消歧。
embedding 配置与索引指纹不匹配时明确降级（返回空，不跨指纹搜索）。
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mikirin.episode_card import CARD_SCHEMA_VERSION
from mikirin.vector_bundle import VectorBundlePublisher
from mikirin.vector_index import EmbeddingClient, EmbeddingConfig, VectorIndex


@dataclass(slots=True)
class SemanticEpisodeHit:
    episode_entity_id: str
    subject_entity_id: str
    subject_name: str
    episode_sort: float
    episode_name: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_entity_id": self.episode_entity_id,
            "subject_entity_id": self.subject_entity_id,
            "subject_name": self.subject_name,
            "episode_sort": self.episode_sort,
            "episode_name": self.episode_name,
            "score": round(self.score, 4),
        }


class SemanticEpisodeSearch:
    """进程级单例：加载配置、索引与 embedding 客户端。"""

    def __init__(self, root: Path, database: Path) -> None:
        self.available = False
        self.fallback_reason = "not-initialized"
        self.index: VectorIndex | None = None
        self.database = database.resolve()
        try:
            cfg = json.loads(
                (root / "data/manifests/runtime_model_config.json").read_text(
                    encoding="utf-8"
                )
            )["roles"]["embedding"]
            self.config = EmbeddingConfig(
                provider=cfg["provider"],
                base_url=cfg["base_url"],
                model=cfg["model"],
                dimensions=int(cfg["dimensions"]),
                distance=cfg.get("distance", "cosine"),
                query_prefix=cfg.get("query_prefix", ""),
                document_prefix=cfg.get("document_prefix", ""),
                normalize=cfg.get("normalize", True),
                input_normalization_version=cfg.get(
                    "input_normalization_version", "unicode-nfc-lf-v1"
                ),
            )
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            self.fallback_reason = f"invalid-embedding-config:{type(error).__name__}"
            return

        try:
            bundle = VectorBundlePublisher(root / "data/vector_index").current(
                expected_embedding_fingerprint=self.config.fingerprint()
            )
        except (KeyError, TypeError, ValueError, OSError, RuntimeError, json.JSONDecodeError) as error:
            self.fallback_reason = f"index-contract-mismatch:{type(error).__name__}"
            return
        if bundle is None:
            self.fallback_reason = "no-published-vector-bundle"
            return
        entry = bundle.indexes.get("episode_cards")
        if not isinstance(entry, dict):
            self.fallback_reason = "episode-card-index-missing"
            return
        try:
            index = VectorIndex(bundle.directory / str(entry["path"]))
        except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            self.fallback_reason = f"episode-card-index-invalid:{type(error).__name__}"
            return
        schema_versions = index.index_contract.get("schema_versions", {})
        if schema_versions.get("episode_card") != CARD_SCHEMA_VERSION:
            index.close()
            self.fallback_reason = "episode-card-schema-mismatch"
            return
        if not self._database_matches_bundle(bundle.source_database):
            index.close()
            self.fallback_reason = "content-database-mismatch"
            return
        key = os.environ.get(str(cfg.get("key_env", "")), "").strip()
        if not key:
            index.close()
            self.fallback_reason = "embedding-api-key-missing"
            return
        self.client = EmbeddingClient(self.config, key)
        self.index = index
        self.available = True
        self.fallback_reason = ""

    def _database_matches_bundle(self, source: dict[str, object]) -> bool:
        expected_build = str(source.get("build_id", "")).strip()
        if not expected_build:
            return False
        try:
            with closing(
                sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True)
            ) as db:
                row = db.execute("SELECT id FROM build LIMIT 1").fetchone()
        except sqlite3.Error:
            return False
        return row is not None and str(row[0]) == expected_build

    def status(self) -> dict[str, object]:
        return {
            "available": self.available,
            "fallback": None if self.available else "fts",
            "reason": self.fallback_reason or None,
            "embedding_fingerprint": getattr(self, "config", None).fingerprint()
            if hasattr(self, "config")
            else None,
        }

    def search(self, query: str, *, top_k: int = 6) -> list[SemanticEpisodeHit]:
        if not self.available or self.index is None:
            return []
        vector = self.client.embed_query(query)
        results = self.index.search(
            vector,
            coarse=500,
            top=top_k * 3,
            object_kinds={"episode_card"},
        )
        hits: list[SemanticEpisodeHit] = []
        seen_subjects: set[str] = set()
        with sqlite3.connect(
            self.database.resolve().as_uri() + "?mode=ro", uri=True
        ) as db:
            db.row_factory = sqlite3.Row
            for _kind, episode_id, score in results:
                row = db.execute(
                    """
                    SELECT ep.subject_entity_id, ep.sort, ep.name, ep.name_cn,
                           COALESCE(NULLIF(s.name_cn,''), s.name) AS subject_name
                    FROM episodes ep JOIN subjects s ON s.entity_id=ep.subject_entity_id
                    WHERE ep.entity_id=?
                    """,
                    (episode_id,),
                ).fetchone()
                if row is None:
                    continue
                hits.append(
                    SemanticEpisodeHit(
                        episode_entity_id=episode_id,
                        subject_entity_id=row["subject_entity_id"],
                        subject_name=row["subject_name"],
                        episode_sort=row["sort"],
                        episode_name=row["name_cn"] or row["name"] or "",
                        score=score,
                    )
                )
                seen_subjects.add(row["subject_entity_id"])
                if len(hits) >= top_k:
                    break
        return hits

    def close(self) -> None:
        if self.index is not None:
            self.index.close()

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterable, Mapping

from mikirin.graph.implicit import (EntityBrief, ImplicitCandidate, aggregate_implicit_edges,
                                    RELATION_VOCAB, SENTIMENT_VOCAB)
from mikirin.p1_build import stable_id


def _brief(db: sqlite3.Connection, entity_id: str) -> EntityBrief:
    row = db.execute(
        "SELECT entity_type,canonical_name FROM entity WHERE id=?", (entity_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown entity: {entity_id}")
    entity_type, name = str(row[0]), str(row[1])
    table = {"subject": "subjects", "character": "characters", "person": "persons"}.get(
        entity_type
    )
    summary = ""
    if table is not None:
        detail = db.execute(f"SELECT summary FROM {table} WHERE entity_id=?", (entity_id,)).fetchone()
        summary = str(detail[0] or "") if detail else ""
    return EntityBrief(entity_id, name, (summary or f"{entity_type}: {name}")[:1_200])


def load_implicit_candidates(
    database: Path,
    *,
    run_id: str,
    pair_limit: int = 500,
    evidence_per_pair: int = 3,
    minimum_author_weight: int = 2,
) -> list[ImplicitCandidate]:
    if pair_limit < 1 or evidence_per_pair < 1 or minimum_author_weight < 0:
        raise ValueError("implicit candidate limits are invalid")
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        pairs = db.execute(
            """
            SELECT src_entity_id,dst_entity_id,
                   count(DISTINCT nullif(trim(author_hash),'')) AS author_weight,
                   count(*) AS evidence_count
            FROM cooccurrence WHERE run_id=?
            GROUP BY src_entity_id,dst_entity_id
            HAVING author_weight>=?
            ORDER BY author_weight DESC,evidence_count DESC,src_entity_id,dst_entity_id
            LIMIT ?
            """,
            (run_id, minimum_author_weight, pair_limit),
        ).fetchall()
        result: list[ImplicitCandidate] = []
        for src, dst, _, _ in pairs:
            entities = (_brief(db, str(src)), _brief(db, str(dst)))
            rows = db.execute(
                """
                SELECT co.id,co.document_id,d.content,co.author_hash,c.source_kind
                FROM cooccurrence co JOIN community_document c ON c.document_id=co.document_id
                JOIN source_document d ON d.id=co.document_id
                WHERE co.run_id=? AND co.src_entity_id=? AND co.dst_entity_id=?
                ORDER BY CASE WHEN nullif(trim(co.author_hash),'') IS NULL THEN 1 ELSE 0 END,
                         length(d.content),co.document_id
                LIMIT ?
                """,
                (run_id, src, dst, evidence_per_pair),
            ).fetchall()
            for candidate_id, document_id, evidence_text, author_hash, source_kind in rows:
                result.append(
                    ImplicitCandidate(
                        candidate_id=str(candidate_id),
                        document_id=str(document_id),
                        text=str(evidence_text),
                        entities=entities,
                        author_hash=str(author_hash or ""),
                        source=str(source_kind),
                    )
                )
    return result


def apply_implicit_results(
    database: Path,
    *,
    run_id: str,
    decisions: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    accepted = 0
    rejected = 0
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        build_row = db.execute("SELECT build_id FROM slang_run WHERE id=?", (run_id,)).fetchone()
        if build_row is None:
            raise KeyError(f"unknown slang/cooccurrence run: {run_id}")
        build_id = str(build_row[0])
        for raw in decisions:
            provenance = raw.get("provenance", {})
            if not isinstance(provenance, Mapping):
                raise ValueError("implicit decision provenance must be an object")
            candidate_id = str(provenance.get("candidate_id", ""))
            row = db.execute(
                """
                SELECT co.document_id,co.src_entity_id,co.dst_entity_id,co.author_hash,
                       d.content,c.source_kind
                FROM cooccurrence co JOIN source_document d ON d.id=co.document_id
                JOIN community_document c ON c.document_id=co.document_id
                WHERE co.id=? AND co.run_id=?
                """,
                (candidate_id, run_id),
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown implicit candidate: {candidate_id}")
            if raw.get("reject") == "reject":
                rejected += 1
                continue
            if raw.get("relation") not in RELATION_VOCAB or raw.get("sentiment") not in SENTIMENT_VOCAB:
                raise ValueError("implicit relation/sentiment is outside the closed vocabulary")
            confidence = raw.get("confidence")
            if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                    or not math.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError("implicit confidence must be finite and in [0,1]")
            src, dst = sorted((str(raw.get("src", "")), str(raw.get("dst", ""))))
            if (src, dst) != (str(row[1]), str(row[2])):
                raise ValueError(f"implicit endpoints are outside Pass A pair: {candidate_id}")
            claim_quote = str(raw.get("claim_quote", ""))
            span = provenance.get("span", {})
            if not isinstance(span, Mapping):
                span = {}
            char_start = int(span.get("start", -1))
            char_end = int(span.get("end", -1))
            if (
                not claim_quote
                or claim_quote not in str(row[4])
                or char_start < 0
                or char_end < char_start
                or str(row[4])[char_start:char_end] != claim_quote
            ):
                raise ValueError(f"implicit claim quote is not in source document: {candidate_id}")
            cache_key = str(provenance.get("cache_key", ""))
            model = str(provenance.get("model", ""))
            if not cache_key or not model:
                raise ValueError("implicit decision is missing model/cache provenance")
            evidence_id = stable_id(
                "implicit-evidence",
                run_id,
                row[0],
                src,
                dst,
                raw.get("relation"),
                claim_quote,
            )
            db.execute(
                """
                INSERT INTO implicit_edge_evidence(
                    id,run_id,document_id,src_entity_id,dst_entity_id,relation,
                    sentiment,confidence,claim_quote,char_start,char_end,author_hash,
                    model_fingerprint,cache_key
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(run_id,document_id,src_entity_id,dst_entity_id,relation,claim_quote)
                DO UPDATE SET sentiment=excluded.sentiment,confidence=excluded.confidence,
                              author_hash=excluded.author_hash,
                              model_fingerprint=excluded.model_fingerprint,
                              cache_key=excluded.cache_key
                """,
                (
                    evidence_id,
                    run_id,
                    row[0],
                    src,
                    dst,
                    str(raw["relation"]),
                    str(raw["sentiment"]),
                    float(raw["confidence"]),
                    claim_quote,
                    char_start,
                    char_end,
                    str(row[3] or ""),
                    model,
                    cache_key,
                ),
            )
            accepted += 1
        observations: list[dict[str, Any]] = []
        for row in db.execute(
            """
            SELECT e.src_entity_id,e.dst_entity_id,e.relation,e.sentiment,e.confidence,
                   e.claim_quote,e.author_hash,c.source_kind,e.document_id,e.cache_key
            FROM implicit_edge_evidence e
            JOIN community_document c ON c.document_id=e.document_id
            WHERE e.run_id=?
            ORDER BY e.src_entity_id,e.dst_entity_id,e.relation,e.sentiment,e.document_id
            """,
            (run_id,),
        ):
            observations.append(
                {
                    "src": row[0],
                    "dst": row[1],
                    "relation": row[2],
                    "sentiment": row[3],
                    "confidence": float(row[4]),
                    "claim_quote": row[5],
                    "author_hash": row[6],
                    "source_kind": row[7],
                    "document_id": row[8],
                    "cache_key": row[9],
                }
            )
        aggregated = aggregate_implicit_edges(observations)
        db.execute("DELETE FROM implicit_edge WHERE build_id=?", (build_id,))
        for edge in aggregated:
            db.execute(
                "INSERT INTO implicit_edge VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    edge["src"],
                    edge["dst"],
                    edge["relation"],
                    edge["sentiment"],
                    edge["weight"],
                    edge["evidence_count"],
                    edge["confidence"],
                    json.dumps(
                        {"attribution": edge["attribution"], "run_id": run_id},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    build_id,
                ),
            )
    return {
        "accepted_evidence": accepted,
        "rejected": rejected,
        "implicit_edges": len(aggregated),
    }


__all__ = ["apply_implicit_results", "load_implicit_candidates"]

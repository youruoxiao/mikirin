from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from mikirin.p1_build import stable_id
from mikirin.slang.arbitration import CommentContext, EntityBrief, SlangCandidate
from mikirin.slang.candidates import normalize_for_matching


ALIAS_POLICY_VERSION = "mined-alias-policy-v1"


@dataclass(frozen=True, slots=True)
class AliasPolicy:
    confirmed_confidence: float = 0.90
    provisional_confidence: float = 0.72
    confirmed_link_score: float = 0.72
    provisional_link_score: float = 0.55
    single_confirmed_confidence: float = 0.97
    single_confirmed_link_score: float = 0.90
    confirmed_author_breadth: int = 20

    def validate(self) -> None:
        for value in (
            self.confirmed_confidence,
            self.provisional_confidence,
            self.confirmed_link_score,
            self.provisional_link_score,
            self.single_confirmed_confidence,
            self.single_confirmed_link_score,
        ):
            if not 0 <= value <= 1:
                raise ValueError("alias policy scores must be in [0,1]")
        if self.confirmed_author_breadth < 0:
            raise ValueError("confirmed_author_breadth must be non-negative")
        if not (self.provisional_confidence <= self.confirmed_confidence <= self.single_confirmed_confidence
                and self.provisional_link_score <= self.confirmed_link_score <= self.single_confirmed_link_score):
            raise ValueError("single-character and confirmed thresholds must not be weaker")


def _entity_brief(db: sqlite3.Connection, entity_id: str) -> EntityBrief:
    row = db.execute(
        "SELECT entity_type,canonical_name FROM entity WHERE id=?", (entity_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"unknown entity: {entity_id}")
    entity_type, name = str(row[0]), str(row[1])
    if entity_type == "subject":
        detail = db.execute(
            "SELECT coalesce(name_cn,name),summary FROM subjects WHERE entity_id=?",
            (entity_id,),
        ).fetchone()
    elif entity_type == "character":
        detail = db.execute(
            "SELECT name,summary FROM characters WHERE entity_id=?", (entity_id,)
        ).fetchone()
    elif entity_type == "person":
        detail = db.execute(
            "SELECT name,summary FROM persons WHERE entity_id=?", (entity_id,)
        ).fetchone()
    else:
        detail = None
    display_name = str(detail[0] or name) if detail else name
    summary = str(detail[1] or "") if detail else ""
    if not summary.strip():
        summary = f"{entity_type}: {display_name}"
    return EntityBrief(entity_id, display_name, summary[:1_200])


def load_arbitration_candidates(
    database: Path,
    *,
    run_id: str,
    limit: int = 500,
    evidence_per_candidate: int = 5,
) -> list[SlangCandidate]:
    if limit < 1 or evidence_per_candidate < 1:
        raise ValueError("limits must be positive")
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        candidates = db.execute(
            """
            SELECT id,candidate_text FROM slang_candidate
            WHERE run_id=? AND status='shortlisted'
              AND EXISTS(
                  SELECT 1 FROM slang_link_candidate l
                  WHERE l.candidate_id=slang_candidate.id AND l.decision='pending'
              )
            ORDER BY score DESC,author_count DESC,document_count DESC,normalized_candidate
            LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
        result: list[SlangCandidate] = []
        for candidate_id, candidate_text in candidates:
            comments: list[CommentContext] = []
            rows = db.execute(
                """
                SELECT ev.document_id,d.content,min(ev.char_start),max(ev.char_end)
                FROM slang_evidence ev JOIN source_document d ON d.id=ev.document_id
                WHERE ev.candidate_id=?
                GROUP BY ev.document_id,d.content
                ORDER BY ev.document_id,min(ev.char_start) LIMIT ?
                """,
                (candidate_id, evidence_per_candidate),
            ).fetchall()
            for document_id, content, start, end in rows:
                content = str(content)
                start, end = int(start), int(end)
                excerpt_start = max(0, start - 300)
                excerpt_end = min(len(content), end + 300)
                comments.append(
                    CommentContext(
                        comment_id=str(document_id),
                        content=content[excerpt_start:excerpt_end],
                        source_offset=excerpt_start,
                        source_digest="sha256:"
                        + hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    )
                )
            entity_ids = [
                str(row[0])
                for row in db.execute(
                    "SELECT entity_id FROM slang_link_candidate WHERE candidate_id=? "
                    "AND decision='pending' ORDER BY rank,entity_id LIMIT 3",
                    (candidate_id,),
                )
            ]
            if comments and entity_ids:
                result.append(
                    SlangCandidate(
                        str(candidate_id),
                        str(candidate_text),
                        tuple(comments),
                        tuple(_entity_brief(db, entity_id) for entity_id in entity_ids),
                    )
                )
    return result


def _decision_status(
    *,
    bucket: str,
    link_score: float,
    confidence: float,
    author_count: int,
    policy: AliasPolicy,
) -> str | None:
    if bucket == "single_char":
        if (
            confidence >= policy.single_confirmed_confidence
            and link_score >= policy.single_confirmed_link_score
            and author_count > policy.confirmed_author_breadth
        ):
            return "confirmed"
        if confidence >= policy.confirmed_confidence and link_score >= policy.confirmed_link_score:
            return "provisional"
        return None
    if (
        confidence >= policy.confirmed_confidence
        and link_score >= policy.confirmed_link_score
        and author_count > policy.confirmed_author_breadth
    ):
        return "confirmed"
    if confidence >= policy.provisional_confidence and link_score >= policy.provisional_link_score:
        return "provisional"
    return None


def _lang(value: str) -> str | None:
    if any("\u4e00" <= char <= "\u9fff" for char in value):
        return "zh"
    if any("\u3040" <= char <= "\u30ff" for char in value):
        return "ja"
    if any(char.isascii() and char.isalpha() for char in value):
        return "en"
    return None


def _refresh_entity_fts(db: sqlite3.Connection, entity_ids: Iterable[str]) -> None:
    for entity_id in sorted(set(entity_ids)):
        db.execute("DELETE FROM entity_fts WHERE entity_id=?", (entity_id,))
        row = db.execute(
            """
            SELECT e.id,e.entity_type,e.canonical_name,coalesce(group_concat(a.alias_text,' '),'')
            FROM entity e LEFT JOIN entity_alias a ON a.entity_id=e.id
              AND (a.source_kind<>'mined' OR a.status IN ('confirmed','provisional'))
            WHERE e.id=? GROUP BY e.id,e.entity_type,e.canonical_name
            """,
            (entity_id,),
        ).fetchone()
        if row is not None:
            db.execute("INSERT INTO entity_fts VALUES(?,?,?,?)", row)


def apply_arbitration_results(
    database: Path,
    *,
    run_id: str,
    decisions: Iterable[Mapping[str, Any]],
    policy: AliasPolicy | None = None,
) -> dict[str, int]:
    config = policy or AliasPolicy()
    config.validate()
    counts = {"confirmed": 0, "provisional": 0, "rejected": 0, "aliases_written": 0}
    touched_entities: set[str] = set()
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        for raw in decisions:
            candidate_id = str(raw.get("candidate_id", ""))
            decision = str(raw.get("decision", ""))
            confidence = float(raw.get("confidence", 0.0))
            if isinstance(raw.get("confidence"), bool) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("decision confidence must be finite and in [0,1]")
            provenance = raw.get("provenance", {})
            if not isinstance(provenance, Mapping):
                raise ValueError("decision provenance must be an object")
            comment_id = str(raw.get("comment_id", ""))
            quote = str(raw.get("quote", ""))
            evidence_row = db.execute(
                """
                SELECT d.content FROM slang_evidence ev
                JOIN source_document d ON d.id=ev.document_id
                WHERE ev.candidate_id=? AND ev.document_id=? LIMIT 1
                """,
                (candidate_id, comment_id),
            ).fetchone()
            source_content = "" if evidence_row is None else str(evidence_row[0])
            expected_digest = "sha256:" + hashlib.sha256(source_content.encode("utf-8")).hexdigest()
            supplied_digest = str(provenance.get("source_digest", ""))
            if (
                not quote.strip()
                or evidence_row is None
                or quote not in source_content
                or (supplied_digest and supplied_digest != expected_digest)
            ):
                raise ValueError(f"arbitration quote is not traceable: {candidate_id}")
            row = db.execute(
                """
                SELECT c.candidate_text,c.normalized_candidate,c.bucket,c.author_count,
                       c.run_id,l.link_score
                FROM slang_candidate c LEFT JOIN slang_link_candidate l
                  ON l.candidate_id=c.id AND l.entity_id=?
                WHERE c.id=?
                """,
                (decision, candidate_id),
            ).fetchone()
            if row is None or str(row[4]) != run_id:
                raise ValueError(f"unknown candidate/decision pair: {candidate_id} -> {decision}")
            if decision != "reject" and row[5] is None:
                raise ValueError(f"decision is outside closed candidate set: {decision}")
            if decision == "reject":
                db.execute(
                    "UPDATE slang_link_candidate SET decision='rejected',decision_confidence=?,"
                    "arbitration_json=? WHERE candidate_id=?",
                    (
                        confidence,
                        json.dumps(raw, ensure_ascii=False, sort_keys=True),
                        candidate_id,
                    ),
                )
                db.execute("UPDATE slang_candidate SET status='rejected' WHERE id=?", (candidate_id,))
                counts["rejected"] += 1
                continue
            candidate_text, normalized, bucket, author_count, _, link_score = row
            status = _decision_status(
                bucket=str(bucket),
                link_score=float(link_score),
                confidence=confidence,
                author_count=int(author_count),
                policy=config,
            )
            db.execute(
                "UPDATE slang_link_candidate SET decision=?,decision_confidence=?,arbitration_json=? "
                "WHERE candidate_id=? AND entity_id=?",
                (
                    status or "rejected",
                    confidence,
                    json.dumps(raw, ensure_ascii=False, sort_keys=True),
                    candidate_id,
                    decision,
                ),
            )
            db.execute(
                "UPDATE slang_link_candidate SET decision='rejected' "
                "WHERE candidate_id=? AND entity_id<>? AND decision='pending'",
                (candidate_id, decision),
            )
            db.execute(
                "UPDATE slang_candidate SET status=? WHERE id=?",
                (status or "rejected", candidate_id),
            )
            if status is None:
                counts["rejected"] += 1
                continue
            alias_id = stable_id("alias-mined", run_id, decision, normalized)
            alias_confidence = min(float(link_score), confidence)
            before = db.total_changes
            db.execute(
                """
                INSERT INTO entity_alias(
                    id,entity_id,alias_text,normalized_alias,lang,source_kind,status,
                    confidence,source_asset_id,build_id,license_id
                )
                SELECT ?,?,?,?,?,'mined',?,?,NULL,r.build_id,'community-derived'
                FROM slang_run r WHERE r.id=?
                ON CONFLICT(entity_id,normalized_alias,source_kind,build_id) DO UPDATE SET
                    alias_text=excluded.alias_text,lang=excluded.lang,status=excluded.status,
                    confidence=excluded.confidence
                """,
                (
                    alias_id,
                    decision,
                    str(candidate_text),
                    normalize_for_matching(str(normalized)),
                    _lang(str(candidate_text)),
                    status,
                    alias_confidence,
                    run_id,
                ),
            )
            if db.total_changes > before:
                counts["aliases_written"] += 1
            counts[status] += 1
            touched_entities.add(decision)
        _refresh_entity_fts(db, touched_entities)
    return counts


__all__ = [
    "ALIAS_POLICY_VERSION",
    "AliasPolicy",
    "apply_arbitration_results",
    "load_arbitration_candidates",
]

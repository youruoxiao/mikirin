"""Deterministic, read-only sources for the bounded semantic index.

This module deliberately contains no embedding or index-writing code.  It turns
the published SQLite rows into stable, typed records that an index builder can
consume without changing the database.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from mikirin.episode_card import CARD_SCHEMA_VERSION, build_episode_cards


VECTOR_SOURCE_SCHEMA_VERSION = "vector-source-v1"
COMMUNITY_CHUNK_SCHEMA_VERSION = "community-chunk-v1"
COMMUNITY_REPORT_SCHEMA_VERSION = "community-report-vector-v1"
DISCUSSION_THEME_SCHEMA_VERSION = "discussion-theme-vector-v1"
COMMUNITY_CHUNK_SOURCE_KINDS = (
    "bgm_review",
    "bgm_ep_comment",
    "bgm_subject_comment",
)
SOURCE_TYPE_ORDER = (
    "episode_card",
    "community_chunk",
    "community_report",
    "discussion_theme",
)


@dataclass(frozen=True, slots=True)
class VectorSource:
    """One deterministic document to be passed to an embedding client."""

    source_type: str
    source_id: str
    text: str
    subject_entity_id: str | None = None
    episode_entity_id: str | None = None
    document_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = VECTOR_SOURCE_SCHEMA_VERSION

    @property
    def kind(self) -> str:
        return self.source_type

    @property
    def id(self) -> str:
        return self.source_id


def _database_uri(database: Path | str) -> str:
    return Path(database).resolve().as_uri() + "?mode=ro"


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _stable_json(value: Any) -> str:
    value = _json_value(value)
    if isinstance(value, (dict, list, tuple)):
        if not value:
            return ""
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _text(value)


def _has_column(db: sqlite3.Connection, table: str, column: str) -> bool:
    return column in {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def _open(database: Path | str) -> sqlite3.Connection:
    db = sqlite3.connect(_database_uri(database), uri=True)
    db.row_factory = sqlite3.Row
    return db


def iter_episode_card_sources(database: Path | str) -> Iterator[VectorSource]:
    """Yield the existing deterministic episode cards, without rebuilding them."""
    for card in build_episode_cards(Path(database)):
        if not _text(card.text):
            continue
        yield VectorSource(
            source_type="episode_card",
            source_id=card.episode_id,
            text=card.text,
            subject_entity_id=card.subject_id,
            episode_entity_id=card.episode_id,
            metadata={"comment_count": card.comment_count},
            schema_version=CARD_SCHEMA_VERSION,
        )


def iter_community_chunk_sources(database: Path | str) -> Iterator[VectorSource]:
    """Yield only the three approved community document-chunk source kinds."""
    with closing(_open(database)) as db:
        header_expr = "c.context_header" if _has_column(db, "document_chunk", "context_header") else "d.title"
        query = f"""SELECT c.id, c.document_id, c.chunk_index, c.content,
                           {header_expr} AS context_header, d.title, d.source,
                           cd.source_kind, cd.subject_entity_id, cd.episode_entity_id
                    FROM document_chunk c
                    JOIN source_document d ON d.id = c.document_id
                    JOIN community_document cd ON cd.document_id = c.document_id
                    WHERE cd.source_kind IN (?, ?, ?)
                    ORDER BY cd.subject_entity_id, cd.episode_entity_id IS NOT NULL,
                             cd.episode_entity_id, cd.source_kind, c.document_id, c.chunk_index, c.id"""
        for row in db.execute(query, COMMUNITY_CHUNK_SOURCE_KINDS):
            content = _text(row["content"])
            header = _text(row["context_header"])
            text = "\n".join(part for part in (header, content) if part)
            if not text:
                continue
            yield VectorSource(
                source_type="community_chunk",
                source_id=row["id"],
                text=text,
                subject_entity_id=row["subject_entity_id"],
                episode_entity_id=row["episode_entity_id"],
                document_id=row["document_id"],
                metadata={
                    "context_header": header,
                    "content": content,
                    "chunk_index": row["chunk_index"],
                    "document_title": row["title"],
                    "document_source": row["source"],
                    "source_kind": row["source_kind"],
                },
                schema_version=COMMUNITY_CHUNK_SCHEMA_VERSION,
            )


def iter_community_report_sources(database: Path | str) -> Iterator[VectorSource]:
    """Yield one canonical text record per published community report."""
    fields = (
        ("summary", "reception_summary"),
        ("strengths", "strengths_json"),
        ("weaknesses", "weaknesses_json"),
        ("discussion_points", "discussion_points_json"),
        ("disagreements", "notable_disagreements_json"),
        ("key_episodes", "key_episodes_json"),
    )
    with closing(_open(database)) as db:
        for row in db.execute("SELECT * FROM community_report ORDER BY subject_entity_id"):
            parts = []
            for label, column in fields:
                value = _stable_json(row[column])
                if value:
                    parts.append(f"{label}: {value}")
            text = "\n".join(parts)
            if not text:
                continue
            yield VectorSource(
                source_type="community_report",
                source_id=row["subject_entity_id"],
                text=text,
                subject_entity_id=row["subject_entity_id"],
                metadata={"report_fields": tuple(label for label, _ in fields if _text(row[_]))},
                schema_version=COMMUNITY_REPORT_SCHEMA_VERSION,
            )


def _quotes(value: Any) -> list[str]:
    parsed = _json_value(value)
    if not isinstance(parsed, list):
        return []
    result = []
    for item in parsed:
        quote = item.get("quote") if isinstance(item, dict) else item
        quote = _text(quote)
        if quote:
            result.append(quote)
        if len(result) == 2:
            break
    return result


def iter_discussion_theme_sources(database: Path | str) -> Iterator[VectorSource]:
    """Yield themes as signposts containing at most two verbatim quotes."""
    with closing(_open(database)) as db:
        for row in db.execute(
            """SELECT id, subject_entity_id, episode_entity_id, theme_text,
                      aspect_kind, quotes_json
               FROM discussion_theme
               ORDER BY subject_entity_id, episode_entity_id IS NOT NULL,
                        episode_entity_id, theme_text, id"""
        ):
            theme = _text(row["theme_text"])
            quotes = _quotes(row["quotes_json"])
            text = "\n".join([f"theme: {theme}"] + [f"quote: {quote}" for quote in quotes]) if theme else ""
            if not _text(text):
                continue
            yield VectorSource(
                source_type="discussion_theme",
                source_id=row["id"],
                text=text,
                subject_entity_id=row["subject_entity_id"],
                episode_entity_id=row["episode_entity_id"],
                metadata={"aspect_kind": row["aspect_kind"], "quotes": tuple(quotes)},
                schema_version=DISCUSSION_THEME_SCHEMA_VERSION,
            )


def iter_vector_sources(database: Path | str) -> Iterator[VectorSource]:
    """Yield all bounded vector sources in a globally stable order."""
    iterators = (
        iter_episode_card_sources(database),
        iter_community_chunk_sources(database),
        iter_community_report_sources(database),
        iter_discussion_theme_sources(database),
    )
    for iterator in iterators:
        yield from iterator


__all__ = [
    "COMMUNITY_CHUNK_SOURCE_KINDS",
    "COMMUNITY_CHUNK_SCHEMA_VERSION",
    "COMMUNITY_REPORT_SCHEMA_VERSION",
    "DISCUSSION_THEME_SCHEMA_VERSION",
    "SOURCE_TYPE_ORDER",
    "VECTOR_SOURCE_SCHEMA_VERSION",
    "VectorSource",
    "iter_community_chunk_sources",
    "iter_community_report_sources",
    "iter_discussion_theme_sources",
    "iter_episode_card_sources",
    "iter_vector_sources",
]

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


USER_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS watch_progress(
    subject_entity_id TEXT PRIMARY KEY,
    episode_entity_id TEXT,
    progress_value REAL NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source_message_id TEXT,
    confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1)
);
CREATE TABLE IF NOT EXISTS memory_note(
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    keywords TEXT NOT NULL DEFAULT '',
    context TEXT NOT NULL DEFAULT '',
    entity_ids_json TEXT NOT NULL DEFAULT '[]',
    valid_from TEXT NOT NULL,
    valid_until TEXT,
    source_session_id TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_link(
    src_id TEXT NOT NULL REFERENCES memory_note(id),
    dst_id TEXT NOT NULL REFERENCES memory_note(id),
    relation TEXT NOT NULL,
    PRIMARY KEY(src_id, dst_id, relation)
);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_note_fts USING fts5(
    note_id UNINDEXED, content, keywords, context, tokenize='trigram'
);
CREATE TABLE IF NOT EXISTS session(
    id TEXT PRIMARY KEY,
    parent_session_id TEXT,
    created_at TEXT NOT NULL,
    jsonl_path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS message(
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES session(id),
    entry_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, entry_id)
);
CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(
    message_id UNINDEXED, session_id UNINDEXED, content, tokenize='trigram'
);
CREATE TABLE IF NOT EXISTS memory_schema_migration(version TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS memory_settlement_job(
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    messages_json TEXT NOT NULL,
    chunk_chars INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending', 'running', 'failed', 'done')),
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    report_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_settlement_chunk(
    job_id TEXT NOT NULL REFERENCES memory_settlement_job(id),
    chunk_index INTEGER NOT NULL,
    facts_json TEXT NOT NULL,
    PRIMARY KEY(job_id, chunk_index)
);
CREATE TABLE IF NOT EXISTS memory_settlement_receipt(
    session_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    note_id TEXT,
    created_at TEXT NOT NULL,
    source_job_id TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(session_id, content_hash)
);
CREATE INDEX IF NOT EXISTS memory_settlement_pending ON memory_settlement_job(status, created_at);
CREATE TRIGGER IF NOT EXISTS memory_note_insert_fts AFTER INSERT ON memory_note BEGIN
    INSERT INTO memory_note_fts(note_id, content, keywords, context)
    VALUES(new.id, new.content, new.keywords, new.context);
END;
CREATE TRIGGER IF NOT EXISTS memory_note_delete_fts AFTER DELETE ON memory_note BEGIN
    DELETE FROM memory_note_fts WHERE note_id=old.id;
END;
CREATE TRIGGER IF NOT EXISTS memory_note_update_fts
AFTER UPDATE OF content, keywords, context ON memory_note BEGIN
    DELETE FROM memory_note_fts WHERE note_id=old.id;
    INSERT INTO memory_note_fts(note_id, content, keywords, context)
    VALUES(new.id, new.content, new.keywords, new.context);
END;
"""


@dataclass(frozen=True, slots=True)
class ProgressRecord:
    subject_entity_id: str
    episode_entity_id: str | None
    progress_value: float
    status: str
    updated_at: str
    source_message_id: str | None
    confidence: float


class UserStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.executescript(USER_SCHEMA)
            db.execute("BEGIN IMMEDIATE")
            receipt_columns = {row[1] for row in db.execute(
                "PRAGMA table_info(memory_settlement_receipt)"
            )}
            if "source_job_id" not in receipt_columns:
                db.execute("ALTER TABLE memory_settlement_receipt "
                           "ADD COLUMN source_job_id TEXT NOT NULL DEFAULT ''")
            if not db.execute(
                "SELECT 1 FROM memory_schema_migration WHERE version='fts-triggers-v1'"
            ).fetchone():
                db.execute("DELETE FROM memory_note_fts")
                db.execute(
                    "INSERT INTO memory_note_fts(note_id, content, keywords, context) "
                    "SELECT id, content, keywords, context FROM memory_note"
                )
                db.execute("INSERT INTO memory_schema_migration VALUES('fts-triggers-v1')")

    def progress(self, subject_entity_id: str) -> ProgressRecord | None:
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute(
                "SELECT subject_entity_id, episode_entity_id, progress_value, status, "
                "updated_at, source_message_id, confidence FROM watch_progress "
                "WHERE subject_entity_id=?",
                (subject_entity_id,),
            ).fetchone()
        return ProgressRecord(*row) if row else None

    def register_session(
        self,
        session_id: str,
        *,
        jsonl_path: Path,
        parent_session_id: str | None = None,
    ) -> None:
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "INSERT OR IGNORE INTO session(id, parent_session_id, created_at, jsonl_path) "
                "VALUES (?, ?, ?, ?)",
                (
                    session_id,
                    parent_session_id,
                    datetime.now(UTC).isoformat(),
                    str(jsonl_path),
                ),
            )

    def mirror_message(
        self,
        *,
        session_id: str,
        entry_id: str,
        role: str,
        content: str,
    ) -> bool:
        message_id = f"{session_id}:{entry_id}"
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute(
                "INSERT OR IGNORE INTO message(id, session_id, entry_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, session_id, entry_id, role, content, now),
            )
            inserted = bool(db.execute("SELECT changes()").fetchone()[0])
            if inserted:
                db.execute(
                    "INSERT INTO message_fts(message_id, session_id, content) VALUES (?, ?, ?)",
                    (message_id, session_id, content),
                )
        return inserted

    def search_messages(self, query: str, *, limit: int = 20) -> list[dict[str, str]]:
        if not query.strip() or limit < 1:
            return []
        phrase = '"' + query.replace('"', '""') + '"'
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute(
                """
                SELECT m.id, m.session_id, m.role, m.content, m.created_at
                FROM message_fts f JOIN message m ON m.id=f.message_id
                WHERE message_fts MATCH ?
                ORDER BY bm25(message_fts) LIMIT ?
                """,
                (phrase, limit),
            ).fetchall()
        return [
            {
                "message_id": row[0],
                "session_id": row[1],
                "role": row[2],
                "content": row[3],
                "created_at": row[4],
            }
            for row in rows
        ]

    def search_memory_notes(
        self, query: str, *, limit: int = 20, include_expired: bool = False,
    ) -> list[dict[str, str]]:
        if not query.strip() or limit < 1:
            return []
        if len(query.strip()) < 3 and not include_expired:
            return self.active_memory_notes(query=query, limit=limit)
        phrase = '"' + query.replace('"', '""') + '"'
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute(
                """
                SELECT n.id, n.content, n.context, n.valid_from, n.valid_until,
                       n.source_session_id
                FROM memory_note_fts f JOIN memory_note n ON n.id=f.note_id
                WHERE memory_note_fts MATCH ? AND (? OR n.valid_until IS NULL)
                  AND n.valid_from <= ?
                ORDER BY bm25(memory_note_fts) LIMIT ?
                """,
                (phrase, include_expired, datetime.now(UTC).isoformat(), limit),
            ).fetchall()
        return [
            {
                "note_id": row[0],
                "content": row[1],
                "context": row[2],
                "valid_from": row[3],
                "valid_until": row[4],
                "source_session_id": row[5],
            }
            for row in rows
        ]

    def active_memory_notes(
        self, *, limit: int = 20, entity_ids: list[str] | None = None,
        query: str | None = None,
    ) -> list[dict]:
        """Read current cards, filtering before LIMIT; supports short/CJK queries."""
        with closing(sqlite3.connect(self.path)) as db:
            return active_memory_notes(db, limit=limit, entity_ids=entity_ids, query=query)

    def surrounding_messages(self, message_id: str, *, radius: int = 3) -> list[dict[str, str]]:
        if radius < 0 or radius > 20:
            raise ValueError("radius must be between 0 and 20")
        with closing(sqlite3.connect(self.path)) as db:
            anchor = db.execute(
                "SELECT session_id FROM message WHERE id=?", (message_id,)
            ).fetchone()
            if anchor is None:
                return []
            rows = db.execute(
                "SELECT id, role, content, created_at FROM message WHERE session_id=? "
                "ORDER BY created_at, rowid",
                (anchor[0],),
            ).fetchall()
        index = next((i for i, row in enumerate(rows) if row[0] == message_id), None)
        if index is None:
            return []
        selected = rows[max(0, index - radius) : index + radius + 1]
        return [
            {"message_id": row[0], "role": row[1], "content": row[2], "created_at": row[3]}
            for row in selected
        ]

    def recent_sessions(self, *, limit: int = 10) -> list[dict[str, str | None]]:
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute(
                """
                SELECT s.id, s.parent_session_id, s.created_at, s.jsonl_path,
                       count(m.id), max(m.created_at)
                FROM session s LEFT JOIN message m ON m.session_id=s.id
                GROUP BY s.id ORDER BY coalesce(max(m.created_at), s.created_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "session_id": row[0],
                "parent_session_id": row[1],
                "created_at": row[2],
                "jsonl_path": row[3],
                "message_count": str(row[4]),
                "last_message_at": row[5],
            }
            for row in rows
        ]

    def all_progress(self) -> list[ProgressRecord]:
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute(
                "SELECT subject_entity_id, episode_entity_id, progress_value, status, "
                "updated_at, source_message_id, confidence FROM watch_progress "
                "ORDER BY updated_at DESC"
            ).fetchall()
        return [ProgressRecord(*row) for row in rows]

    def update_progress(
        self,
        *,
        subject_entity_id: str,
        episode_entity_id: str | None,
        progress_value: float,
        status: str,
        source_message_id: str | None,
        confidence: float,
        source_session_id: str | None = None,
    ) -> tuple[ProgressRecord | None, ProgressRecord]:
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if progress_value < 0:
            raise ValueError("progress_value cannot be negative")
        now = datetime.now(UTC).isoformat()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT subject_entity_id, episode_entity_id, progress_value, status, "
                "updated_at, source_message_id, confidence FROM watch_progress "
                "WHERE subject_entity_id=?",
                (subject_entity_id,),
            ).fetchone()
            previous = ProgressRecord(*row) if row else None
            if previous is not None and confidence < previous.confidence:
                raise ValueError(
                    "lower-confidence progress cannot silently overwrite existing progress; ask the user"
                )
            if previous is not None:
                db.execute(
                    "UPDATE memory_note SET valid_until=? WHERE valid_until IS NULL "
                    "AND json_extract(entity_ids_json, '$[0]')=? AND keywords='watch_progress'",
                    (now, subject_entity_id),
                )
            db.execute(
                """
                INSERT INTO watch_progress(
                    subject_entity_id, episode_entity_id, progress_value, status,
                    updated_at, source_message_id, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(subject_entity_id) DO UPDATE SET
                    episode_entity_id=excluded.episode_entity_id,
                    progress_value=excluded.progress_value,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    source_message_id=excluded.source_message_id,
                    confidence=excluded.confidence
                """,
                (
                    subject_entity_id,
                    episode_entity_id,
                    progress_value,
                    status,
                    now,
                    source_message_id,
                    confidence,
                ),
            )
            note_id = "memory:" + uuid.uuid4().hex
            db.execute(
                """
                INSERT INTO memory_note(
                    id, content, keywords, context, entity_ids_json,
                    valid_from, source_session_id, created_at
                ) VALUES (?, ?, 'watch_progress', 'structured progress update', ?, ?, ?, ?)
                """,
                (
                    note_id,
                    f"观看进度为 {progress_value:g}，状态 {status}",
                    json.dumps([subject_entity_id], ensure_ascii=False),
                    now,
                    source_session_id,
                    now,
                ),
            )
        current = ProgressRecord(
            subject_entity_id,
            episode_entity_id,
            progress_value,
            status,
            now,
            source_message_id,
            confidence,
        )
        return previous, current


def active_memory_notes(
    db: sqlite3.Connection, *, limit: int = 20,
    entity_ids: list[str] | None = None, query: str | None = None,
    include_progress: bool = True,
) -> list[dict]:
    """Read-only helper usable by prompt building without schema initialization."""
    if limit < 1:
        return []
    clauses = ["valid_until IS NULL", "valid_from <= ?"]
    params: list = [datetime.now(UTC).isoformat()]
    if not include_progress:
        clauses.append("keywords!='watch_progress'")
    if entity_ids is not None:
        clauses.append("EXISTS (SELECT 1 FROM json_each(entity_ids_json) "
                       "WHERE value IN (SELECT value FROM json_each(?)))")
        params.append(json.dumps(entity_ids))
    if query and query.strip():
        terms = query.split()
        clauses.append("(" + " OR ".join(
            "instr(lower(content || ' ' || keywords), lower(?)) > 0" for _ in terms
        ) + ")")
        params.extend(terms)
    cursor = db.execute(
        "SELECT id AS note_id, content, keywords, context, entity_ids_json, "
        "valid_from, valid_until, source_session_id, created_at FROM memory_note WHERE "
        + " AND ".join(clauses) + " ORDER BY created_at DESC, id LIMIT ?",
        (*params, limit),
    )
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor]

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class TaskState(StrEnum):
    DISCOVERED = "DISCOVERED"
    FETCHED = "FETCHED"
    PARSED = "PARSED"
    VALIDATED = "VALIDATED"
    LINKED = "LINKED"
    INDEXED = "INDEXED"
    PUBLISHED = "PUBLISHED"
    REJECTED = "REJECTED"


_NEXT_STATE: dict[TaskState, TaskState] = {
    TaskState.DISCOVERED: TaskState.FETCHED,
    TaskState.FETCHED: TaskState.PARSED,
    TaskState.PARSED: TaskState.VALIDATED,
    TaskState.VALIDATED: TaskState.LINKED,
    TaskState.LINKED: TaskState.INDEXED,
    TaskState.INDEXED: TaskState.PUBLISHED,
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class PipelineTask:
    task_key: str
    source: str
    state: TaskState
    input_version: str
    processor_version: str
    retryable: bool
    watermark: str | None


class PipelineStore:
    """Mutable build-control state, intentionally outside the published content DB."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def initialize(self) -> None:
        with closing(self.connect()) as db, db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS pipeline_task (
                    task_key TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    state TEXT NOT NULL,
                    input_version TEXT NOT NULL,
                    output_version TEXT,
                    processor_version TEXT NOT NULL,
                    success_count INTEGER NOT NULL DEFAULT 0 CHECK(success_count >= 0),
                    failure_count INTEGER NOT NULL DEFAULT 0 CHECK(failure_count >= 0),
                    rejected_count INTEGER NOT NULL DEFAULT 0 CHECK(rejected_count >= 0),
                    watermark TEXT,
                    retryable INTEGER NOT NULL DEFAULT 1 CHECK(retryable IN (0, 1)),
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pipeline_transition (
                    id INTEGER PRIMARY KEY,
                    task_key TEXT NOT NULL REFERENCES pipeline_task(task_key),
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )

    def discover(
        self,
        *,
        task_key: str,
        source: str,
        input_version: str,
        processor_version: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        now = utc_now()
        with closing(self.connect()) as db, db:
            cursor = db.execute(
                """
                INSERT OR IGNORE INTO pipeline_task (
                    task_key, source, state, input_version, processor_version,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_key,
                    source,
                    TaskState.DISCOVERED,
                    input_version,
                    processor_version,
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
            if cursor.rowcount:
                db.execute(
                    """
                    INSERT INTO pipeline_transition(task_key, from_state, to_state, recorded_at)
                    VALUES (?, NULL, ?, ?)
                    """,
                    (task_key, TaskState.DISCOVERED, now),
                )
                return True
            return False

    def transition(
        self,
        task_key: str,
        to_state: TaskState,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now()
        with closing(self.connect()) as db, db:
            row = db.execute(
                "SELECT state FROM pipeline_task WHERE task_key = ?", (task_key,)
            ).fetchone()
            if row is None:
                raise KeyError(f"未知管线任务: {task_key}")
            current = TaskState(row["state"])
            allowed = to_state is TaskState.REJECTED or _NEXT_STATE.get(current) is to_state
            if not allowed:
                raise ValueError(f"非法状态跳转: {current} -> {to_state}")
            db.execute(
                "UPDATE pipeline_task SET state = ?, updated_at = ? WHERE task_key = ?",
                (to_state, now, task_key),
            )
            db.execute(
                """
                INSERT INTO pipeline_transition(
                    task_key, from_state, to_state, recorded_at, details_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    task_key,
                    current,
                    to_state,
                    now,
                    json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                ),
            )

    def list_tasks(self) -> list[PipelineTask]:
        with closing(self.connect()) as db:
            rows = db.execute(
                """
                SELECT task_key, source, state, input_version, processor_version,
                       retryable, watermark
                FROM pipeline_task ORDER BY created_at, task_key
                """
            ).fetchall()
        return [
            PipelineTask(
                task_key=row["task_key"],
                source=row["source"],
                state=TaskState(row["state"]),
                input_version=row["input_version"],
                processor_version=row["processor_version"],
                retryable=bool(row["retryable"]),
                watermark=row["watermark"],
            )
            for row in rows
        ]

    def get_task(self, task_key: str) -> PipelineTask | None:
        with closing(self.connect()) as db:
            row = db.execute(
                """
                SELECT task_key, source, state, input_version, processor_version,
                       retryable, watermark
                FROM pipeline_task WHERE task_key = ?
                """,
                (task_key,),
            ).fetchone()
        if row is None:
            return None
        return PipelineTask(
            task_key=row["task_key"],
            source=row["source"],
            state=TaskState(row["state"]),
            input_version=row["input_version"],
            processor_version=row["processor_version"],
            retryable=bool(row["retryable"]),
            watermark=row["watermark"],
        )

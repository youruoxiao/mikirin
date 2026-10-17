from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mikirin.agent.types import ToolExecutionResult, ToolUpdateSink
from mikirin.memory_files import BudgetedTextFile, TextOperation
from mikirin.prompts import MEMORY_CHARACTER_LIMIT, USER_CHARACTER_LIMIT
from mikirin.query import KnowledgeQueries
from mikirin.sources.bgm_aliases import normalize_alias
from mikirin.user_store import UserStore


@dataclass(frozen=True, slots=True)
class KnowledgeSearchTool:
    database: Path
    semantic: Any = None
    name: str = "search_anime"
    description: str = (
        "Search the local anime knowledge base. Returns entity facts, related works, and "
        "entity-anchored community text. "
        "Use this before factual claims about anime, characters, voice actors, staff, or scenes."
    )
    execution_mode: str = "parallel"
    retriever: Any = None
    retrieval_mode: str = "evidence"
    focus: Any = None

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Anime-domain search text"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        tool_call_id: str,
        arguments: Mapping[str, Any],
        update: ToolUpdateSink,
    ) -> ToolExecutionResult:
        query = str(arguments["query"]).strip()
        if not query:
            raise ValueError("query cannot be empty")
        limit = int(arguments.get("limit", 8))
        await update(f"searching local knowledge base: {query}")
        if self.retriever is not None:
            context = await asyncio.to_thread(
                self.retriever.search, query, limit=limit,
                mode=self.retrieval_mode, focus=self.focus,
            )
        else:
            context = await asyncio.to_thread(
                KnowledgeQueries.search_anime_context,
                self.database, query, limit=limit, semantic=self.semantic,
            )
        return ToolExecutionResult(
            json.dumps(
                context,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            details={
                "hit_count": len(context["entities"]),
                "document_count": len(context["documents"]),
            },
        )


@dataclass(frozen=True, slots=True)
class DatabaseQueryTool:
    database: Path
    name: str = "query_db"
    description: str = (
        "Run a read-only predefined knowledge query after search_anime resolves an entity id. "
        "Supported intents: subject_overview, staff, cast, episodes, related_subjects, "
        "person_filmography, character_appearances. Free-form SQL is never accepted."
    )
    execution_mode: str = "parallel"

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": [
                        "subject_overview", "staff", "cast", "episodes",
                        "related_subjects", "person_filmography", "character_appearances",
                    ],
                },
                "entity_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["intent", "entity_id"],
            "additionalProperties": False,
        }

    def _query(self, intent: str, entity_id: str, limit: int) -> dict[str, Any]:
        with closing(sqlite3.connect(self.database)) as db:
            entity = db.execute(
                "SELECT entity_type, canonical_name FROM entity WHERE id=?", (entity_id,)
            ).fetchone()
            if entity is None:
                raise ValueError(f"unknown entity_id: {entity_id}")
            entity_type, canonical_name = str(entity[0]), str(entity[1])
            rows: list[dict[str, Any]]
            if intent == "subject_overview":
                if entity_type != "subject":
                    raise ValueError("subject_overview requires a subject entity")
                row = db.execute(
                    """
                    SELECT s.name, s.name_cn, s.summary, s.date, s.platform, s.tags_json,
                           sc.score, sc.rank, sc.favorite_counts_json
                    FROM subjects s
                    LEFT JOIN score_snapshot sc ON sc.subject_entity_id=s.entity_id
                    WHERE s.entity_id=? ORDER BY sc.rowid DESC LIMIT 1
                    """,
                    (entity_id,),
                ).fetchone()
                rows = [] if row is None else [{
                    "name": row[0], "name_cn": row[1], "summary": row[2],
                    "date": row[3], "platform": row[4],
                    "tags": json.loads(row[5] or "[]"), "score": row[6], "rank": row[7],
                    "favorite_counts": json.loads(row[8] or "{}"),
                }]
            elif intent == "staff":
                if entity_type != "subject":
                    raise ValueError("staff requires a subject entity")
                values = db.execute(
                    """
                    SELECT r.person_id, e.canonical_name, r.position, r.appear_eps_json
                    FROM rel_subject_person r JOIN entity e ON e.id=r.person_id
                    WHERE r.subject_id=? ORDER BY r.position, e.canonical_name LIMIT ?
                    """,
                    (entity_id, limit),
                ).fetchall()
                rows = [{"person_entity_id": r[0], "name": r[1], "position": r[2],
                         "appear_episodes": json.loads(r[3]) if r[3] else None} for r in values]
            elif intent == "cast":
                if entity_type != "subject":
                    raise ValueError("cast requires a subject entity")
                values = db.execute(
                    """
                    SELECT rc.character_id, ce.canonical_name, rc.role_type,
                           rp.person_id, pe.canonical_name
                    FROM rel_subject_character rc
                    JOIN entity ce ON ce.id=rc.character_id
                    LEFT JOIN rel_person_character rp
                      ON rp.subject_id=rc.subject_id AND rp.character_id=rc.character_id
                    LEFT JOIN entity pe ON pe.id=rp.person_id
                    WHERE rc.subject_id=?
                    ORDER BY rc.role_type, ce.canonical_name, pe.canonical_name LIMIT ?
                    """,
                    (entity_id, limit),
                ).fetchall()
                rows = [{
                    "character_entity_id": r[0], "character": r[1], "role_type": r[2],
                    "person_entity_id": r[3], "voice_actor": r[4],
                } for r in values]
            elif intent == "episodes":
                if entity_type != "subject":
                    raise ValueError("episodes requires a subject entity")
                values = db.execute(
                    """
                    SELECT entity_id, sort, type, name, name_cn, airdate, description
                    FROM episodes WHERE subject_entity_id=?
                    ORDER BY type, sort LIMIT ?
                    """,
                    (entity_id, limit),
                ).fetchall()
                rows = [{
                    "episode_entity_id": r[0], "sort": r[1], "type": r[2],
                    "name": r[3], "name_cn": r[4], "airdate": r[5],
                    "description": r[6],
                } for r in values]
            elif intent == "related_subjects":
                if entity_type != "subject":
                    raise ValueError("related_subjects requires a subject entity")
                values = db.execute(
                    """
                    SELECT r.related_subject_id, e.canonical_name, r.relation_type, r.rel_order
                    FROM rel_subject_subject r JOIN entity e ON e.id=r.related_subject_id
                    WHERE r.subject_id=? ORDER BY r.relation_type, r.rel_order LIMIT ?
                    """,
                    (entity_id, limit),
                ).fetchall()
                rows = [{"subject_entity_id": r[0], "name": r[1],
                         "relation_type": r[2], "order": r[3]} for r in values]
            elif intent == "person_filmography":
                if entity_type != "person":
                    raise ValueError("person_filmography requires a person entity")
                values = db.execute(
                    """
                    SELECT DISTINCT r.subject_id, e.canonical_name, r.position
                    FROM rel_subject_person r JOIN entity e ON e.id=r.subject_id
                    WHERE r.person_id=? ORDER BY e.canonical_name LIMIT ?
                    """,
                    (entity_id, limit),
                ).fetchall()
                rows = [{"subject_entity_id": r[0], "name": r[1], "position": r[2]}
                        for r in values]
            elif intent == "character_appearances":
                if entity_type != "character":
                    raise ValueError("character_appearances requires a character entity")
                values = db.execute(
                    """
                    SELECT r.subject_id, e.canonical_name, r.role_type
                    FROM rel_subject_character r JOIN entity e ON e.id=r.subject_id
                    WHERE r.character_id=? ORDER BY r.role_type, e.canonical_name LIMIT ?
                    """,
                    (entity_id, limit),
                ).fetchall()
                rows = [{"subject_entity_id": r[0], "name": r[1], "role_type": r[2]}
                        for r in values]
            else:
                raise ValueError(f"unsupported query intent: {intent}")
        return {
            "intent": intent,
            "entity_id": entity_id,
            "entity_type": entity_type,
            "canonical_name": canonical_name,
            "rows": rows,
        }

    async def execute(
        self,
        tool_call_id: str,
        arguments: Mapping[str, Any],
        update: ToolUpdateSink,
    ) -> ToolExecutionResult:
        intent = str(arguments["intent"])
        entity_id = str(arguments["entity_id"]).strip()
        limit = int(arguments.get("limit", 20))
        await update(f"running predefined query: {intent}")
        result = await asyncio.to_thread(self._query, intent, entity_id, limit)
        return ToolExecutionResult(
            json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            details={"intent": intent, "row_count": len(result["rows"])},
        )


@dataclass(frozen=True, slots=True)
class MemoryWriteTool:
    user_path: Path
    memory_path: Path
    name: str = "write_memory"
    description: str = (
        "Atomically edit the bounded user profile or hot-memory file. Use only for explicit, "
        "durable user facts. Batch operations are all-or-nothing."
    )
    execution_mode: str = "sequential"
    _user: BudgetedTextFile = field(init=False, repr=False, compare=False)
    _memory: BudgetedTextFile = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_user", BudgetedTextFile(self.user_path, limit=USER_CHARACTER_LIMIT, initial_text="# 用户画像\n")
        )
        object.__setattr__(
            self, "_memory", BudgetedTextFile(self.memory_path, limit=MEMORY_CHARACTER_LIMIT, initial_text="# 热记忆\n")
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "user or memory"},
                "operations": {"type": "array", "description": "add/replace/remove operations"},
            },
            "required": ["target", "operations"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        tool_call_id: str,
        arguments: Mapping[str, Any],
        update: ToolUpdateSink,
    ) -> ToolExecutionResult:
        target = str(arguments["target"])
        if target not in {"user", "memory"}:
            raise ValueError("target must be user or memory")
        raw_operations = arguments["operations"]
        if not isinstance(raw_operations, list):
            raise ValueError("operations must be an array")
        operations = []
        for raw in raw_operations:
            if not isinstance(raw, dict):
                raise ValueError("each operation must be an object")
            operation = TextOperation(
                op=str(raw.get("op", "")),  # type: ignore[arg-type]
                text=str(raw.get("text", "")),
                old=str(raw.get("old", "")),
                new=str(raw.get("new", "")),
            )
            operations.append(operation)
        await update(f"validating {len(operations)} memory operations")
        editor = self._user if target == "user" else self._memory
        report = await asyncio.to_thread(editor.apply_batch, operations)
        return ToolExecutionResult(
            json.dumps(
                {
                    "target": target,
                    "characters": report.characters,
                    "remaining": report.remaining,
                    "operations": report.operations,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            details={"target": target, "remaining": report.remaining},
        )


@dataclass(frozen=True, slots=True)
class ProgressUpdateTool:
    content_database: Path
    user_store: UserStore
    session_id: str
    name: str = "update_progress"
    description: str = (
        "Update the user's authoritative anime watching progress in user.db. "
        "Use only when the user explicitly states or confirms a progress value."
    )
    execution_mode: str = "sequential"

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "episode": {"type": "number", "minimum": 0},
                "status": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["subject", "episode"],
            "additionalProperties": False,
        }

    def _resolve(self, subject_query: str, episode_sort: float) -> tuple[str, str | None, str]:
        normalized = normalize_alias(subject_query)
        with closing(sqlite3.connect(self.content_database)) as db:
            subjects = db.execute(
                """
                SELECT DISTINCT e.id, e.canonical_name
                FROM entity_alias a JOIN entity e ON e.id=a.entity_id
                WHERE a.normalized_alias=? AND e.entity_type='subject'
                  AND a.status NOT IN ('rejected', 'expired', 'invalidated')
                  AND (a.source_kind!='mined' OR a.status IN ('confirmed', 'provisional'))
                ORDER BY e.id LIMIT 3
                """,
                (normalized,),
            ).fetchall()
            if not subjects:
                subjects = db.execute(
                    "SELECT id, canonical_name FROM entity WHERE canonical_name=? "
                    "AND entity_type='subject' LIMIT 3",
                    (subject_query,),
                ).fetchall()
            if len(subjects) != 1:
                raise ValueError(
                    "subject must resolve to exactly one work; search first and use its canonical name"
                )
            subject_id, canonical_name = subjects[0]
            episode = db.execute(
                "SELECT entity_id FROM episodes WHERE subject_entity_id=? AND type=0 AND sort=? "
                "ORDER BY entity_id LIMIT 1",
                (subject_id, episode_sort),
            ).fetchone()
        return str(subject_id), str(episode[0]) if episode else None, str(canonical_name)

    async def execute(
        self,
        tool_call_id: str,
        arguments: Mapping[str, Any],
        update: ToolUpdateSink,
    ) -> ToolExecutionResult:
        subject_query = str(arguments["subject"]).strip()
        episode_sort = float(arguments["episode"])
        status = str(arguments.get("status", "watching"))
        if status not in {"watching", "completed", "on_hold", "dropped", "planned"}:
            raise ValueError("invalid watch status")
        confidence = float(arguments.get("confidence", 1.0))
        await update(f"resolving progress target: {subject_query}")
        subject_id, episode_id, canonical_name = await asyncio.to_thread(
            self._resolve, subject_query, episode_sort
        )
        previous, current = await asyncio.to_thread(
            self.user_store.update_progress,
            subject_entity_id=subject_id,
            episode_entity_id=episode_id,
            progress_value=episode_sort,
            status=status,
            source_message_id=tool_call_id,
            confidence=confidence,
            source_session_id=self.session_id,
        )
        return ToolExecutionResult(
            json.dumps(
                {
                    "subject": canonical_name,
                    "subject_entity_id": subject_id,
                    "episode_entity_id": episode_id,
                    "previous_progress": previous.progress_value if previous else None,
                    "progress": current.progress_value,
                    "status": current.status,
                    "confidence": current.confidence,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            details={"subject_entity_id": subject_id, "progress": current.progress_value},
        )


@dataclass(frozen=True, slots=True)
class RecallMemoryTool:
    user_store: UserStore
    name: str = "recall_memory"
    description: str = (
        "Recall cross-session user history. Modes: discovery searches messages and memory cards; "
        "anchor restores context around a message id; recent lists recent sessions."
    )
    execution_mode: str = "parallel"

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {"type": "string"},
                "query": {"type": "string"},
                "message_id": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                "radius": {"type": "integer", "minimum": 0, "maximum": 20},
            },
            "required": ["mode"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        tool_call_id: str,
        arguments: Mapping[str, Any],
        update: ToolUpdateSink,
    ) -> ToolExecutionResult:
        mode = str(arguments["mode"]).lower()
        limit = int(arguments.get("limit", 10))
        await update(f"recalling memory in {mode} mode")
        if mode == "discovery":
            query = str(arguments.get("query", "")).strip()
            if not query:
                raise ValueError("discovery mode requires query")
            messages, notes = await asyncio.gather(
                asyncio.to_thread(self.user_store.search_messages, query, limit=limit),
                asyncio.to_thread(self.user_store.search_memory_notes, query, limit=limit),
            )
            payload = {"mode": mode, "messages": messages, "memory_notes": notes}
        elif mode == "anchor":
            message_id = str(arguments.get("message_id", "")).strip()
            if not message_id:
                raise ValueError("anchor mode requires message_id")
            payload = {
                "mode": mode,
                "messages": await asyncio.to_thread(
                    self.user_store.surrounding_messages,
                    message_id,
                    radius=int(arguments.get("radius", 3)),
                ),
            }
        elif mode == "recent":
            payload = {
                "mode": mode,
                "sessions": await asyncio.to_thread(self.user_store.recent_sessions, limit=limit),
            }
        else:
            raise ValueError("mode must be discovery, anchor, or recent")
        return ToolExecutionResult(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            details={"mode": mode},
        )

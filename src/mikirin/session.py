from __future__ import annotations

import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mikirin.agent.types import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


_SESSION_ID = re.compile(r"^[0-9A-Za-z._-]+$")


def message_to_dict(message: AgentMessage) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool_result",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "content": message.content,
            "is_error": message.is_error,
            "details": dict(message.details),
        }
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": part.text}
            if isinstance(part, TextContent)
            else {
                "type": "tool_call",
                "id": part.id,
                "name": part.name,
                "arguments": dict(part.arguments),
            }
            for part in message.content
        ],
        "stop_reason": message.stop_reason,
        "error_message": message.error_message,
    }


def message_from_dict(payload: dict[str, Any]) -> AgentMessage:
    role = payload.get("role")
    if role == "user":
        return UserMessage(str(payload.get("content", "")))
    if role == "tool_result":
        return ToolResultMessage(
            tool_call_id=str(payload.get("tool_call_id", "")),
            tool_name=str(payload.get("tool_name", "")),
            content=str(payload.get("content", "")),
            is_error=bool(payload.get("is_error", False)),
            details=payload.get("details") if isinstance(payload.get("details"), dict) else {},
        )
    if role != "assistant":
        raise ValueError(f"unknown message role: {role!r}")
    content = []
    for part in payload.get("content", []):
        if part.get("type") == "text":
            content.append(TextContent(str(part.get("text", ""))))
        elif part.get("type") == "tool_call":
            arguments = part.get("arguments")
            content.append(ToolCall(
                id=str(part.get("id", "")),
                name=str(part.get("name", "")),
                arguments=arguments if isinstance(arguments, dict) else {},
            ))
    return AssistantMessage(
        tuple(content),
        stop_reason=payload.get("stop_reason", "stop"),
        error_message=payload.get("error_message"),
    )


class SessionStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def create(self, *, metadata: dict[str, Any] | None = None) -> str:
        session_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:10]
        self.append(session_id, "session_start", metadata or {})
        return session_id

    def path(self, session_id: str) -> Path:
        if not _SESSION_ID.fullmatch(session_id):
            raise ValueError("invalid session id")
        return self.directory / f"{session_id}.jsonl"

    def append(self, session_id: str, entry_type: str, payload: dict[str, Any]) -> str:
        path = self.path(session_id)
        entry_id = uuid.uuid4().hex
        entry = {
            "entry_id": entry_id,
            "session_id": session_id,
            "type": entry_type,
            "created_at": datetime.now(UTC).isoformat(),
            "payload": payload,
        }
        encoded = json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return entry_id

    def append_message(self, session_id: str, message: AgentMessage) -> str:
        return self.append(session_id, message.role, message_to_dict(message))

    def entries(self, session_id: str) -> list[dict[str, Any]]:
        path = self.path(session_id)
        if not path.exists():
            raise FileNotFoundError(path)
        entries = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid session JSONL at line {line_number}") from error
                if entry.get("session_id") != session_id:
                    raise ValueError(f"session id mismatch at line {line_number}")
                entries.append(entry)
        return entries

    def replay_messages(self, session_id: str) -> list[AgentMessage]:
        messages = []
        for entry in self.entries(session_id):
            if entry.get("type") in {"user", "assistant", "tool_result"}:
                messages.append(message_from_dict(entry["payload"]))
        return messages

    def latest_entry(self, session_id: str, entry_type: str) -> dict[str, Any] | None:
        for entry in reversed(self.entries(session_id)):
            if entry.get("type") == entry_type:
                return entry
        return None

    def latest(self) -> str | None:
        paths = sorted(self.directory.glob("*.jsonl"), key=lambda path: path.stat().st_mtime)
        return paths[-1].stem if paths else None

    def fork(self, session_id: str) -> str:
        source_entries = self.entries(session_id)
        fork_id = self.create(metadata={"forked_from": session_id})
        for entry in source_entries:
            if entry.get("type") not in {"user", "assistant", "tool_result", "compaction"}:
                continue
            payload = dict(entry.get("payload") or {})
            payload["origin_entry_id"] = entry.get("entry_id")
            self.append(fork_id, str(entry["type"]), payload)
        return fork_id

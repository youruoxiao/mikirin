from __future__ import annotations

import math
from dataclasses import dataclass

from mikirin.agent.types import AgentMessage, AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from mikirin.prompts import sanitize_context


def estimate_text_tokens(text: str) -> int:
    cjk = sum(
        1
        for char in text
        if "\u3400" <= char <= "\u9fff" or "\u3040" <= char <= "\u30ff"
    )
    return cjk + math.ceil((len(text) - cjk) / 4)


def message_text(message: AgentMessage) -> str:
    if isinstance(message, UserMessage):
        return f"user: {message.content}"
    if isinstance(message, ToolResultMessage):
        return f"tool {message.tool_name}: {message.content}"
    pieces = []
    for part in message.content:
        if isinstance(part, TextContent):
            pieces.append(part.text)
        elif isinstance(part, ToolCall):
            pieces.append(f"tool_call {part.name}: {dict(part.arguments)!r}")
    return "assistant: " + "\n".join(pieces)


def estimate_context_tokens(system_prompt: str, messages: list[AgentMessage]) -> int:
    return estimate_text_tokens(system_prompt) + sum(
        estimate_text_tokens(message_text(message)) + 6 for message in messages
    )


def recent_turn_cutoff(messages: list[AgentMessage], keep_recent_turns: int) -> int:
    if keep_recent_turns < 1:
        raise ValueError("keep_recent_turns must be positive")
    user_indices = [index for index, message in enumerate(messages) if isinstance(message, UserMessage)]
    if len(user_indices) <= keep_recent_turns:
        return 0
    return user_indices[-keep_recent_turns]


def serialize_conversation(messages: list[AgentMessage]) -> str:
    return "\n\n".join(message_text(message) for message in messages)


def compaction_prompt(messages: list[AgentMessage], previous_summary: str | None) -> str:
    conversation = serialize_conversation(messages)
    if previous_summary:
        previous = (
            "<previous-summary>\n"
            f"{sanitize_context(previous_summary)}\n"
            "</previous-summary>\n\n"
        )
        instruction = (
            "更新旧摘要，把新对话合并进去。PRESERVE all existing information that remains "
            "useful, especially user preferences, viewing progress, unresolved questions, decisions, "
            "tool evidence, and uncertainty."
        )
    else:
        previous = ""
        instruction = (
            "为早期对话生成忠实的滚动摘要，保留用户偏好、观看进度、未解决问题、决定、"
            "工具证据和不确定性。"
        )
    return (
        f"{instruction}\n只输出摘要正文，不输出标签或解释。\n\n"
        f"{previous}<conversation>\n{conversation}\n</conversation>"
    )


@dataclass(frozen=True, slots=True)
class CompactionRecord:
    summary: str
    removed_messages: int
    compacted_message_count: int
    preserved_messages: int
    estimated_tokens_before: int

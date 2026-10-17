from __future__ import annotations

import json
from typing import Any

from mikirin.agent.types import AgentMessage, AssistantMessage, TextContent, ToolCall, ToolResultMessage


def convert_to_chat_messages(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "user":
            converted.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            text = "".join(
                part.text for part in message.content if isinstance(part, TextContent)
            )
            tool_calls = [part for part in message.content if isinstance(part, ToolCall)]
            item: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                dict(call.arguments), ensure_ascii=False, separators=(",", ":")
                            ),
                        },
                    }
                    for call in tool_calls
                ]
            converted.append(item)
        elif isinstance(message, ToolResultMessage):
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "name": message.tool_name,
                    "content": message.content,
                }
            )
    return converted


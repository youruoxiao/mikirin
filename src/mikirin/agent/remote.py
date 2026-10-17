from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

from mikirin.agent.types import (
    AgentTool,
    AssistantMessage,
    AssistantStreamEvent,
    TextContent,
    ToolCall,
)


ApiStyle = Literal["responses", "chat-completions"]


def _validate_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid API base URL: {base_url!r}")
    return base_url.rstrip("/")


@dataclass(slots=True)
class RemoteAgentProvider:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    api_style: ApiStyle = "chat-completions"
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        self.base_url = _validate_base_url(self.base_url)
        self.model = self.model.strip()
        if not self.model or not self.api_key.strip():
            raise ValueError("model and api_key cannot be empty")

    async def stream(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: tuple[AgentTool, ...],
    ) -> AsyncIterator[AssistantStreamEvent]:
        yield AssistantStreamEvent("start", AssistantMessage(()))
        try:
            message = await asyncio.to_thread(
                self._complete, system_prompt=system_prompt, messages=messages, tools=tools
            )
        except Exception as error:
            message = AssistantMessage(
                (), stop_reason="error", error_message=f"{type(error).__name__}: {error}"
            )
        yield AssistantStreamEvent("done", message)

    def _complete(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: tuple[AgentTool, ...],
    ) -> AssistantMessage:
        if self.api_style == "responses":
            endpoint = f"{self.base_url}/responses"
            body = {
                "model": self.model,
                "instructions": system_prompt,
                "input": self._responses_input(messages),
            }
            if tools:
                body["tools"] = [self._responses_tool(tool) for tool in tools]
        else:
            endpoint = f"{self.base_url}/chat/completions"
            body = {
                "model": self.model,
                "messages": [{"role": "system", "content": system_prompt}, *messages],
                "temperature": 0.8,
                "stream": False,
            }
            if tools:
                body["tools"] = [self._chat_tool(tool) for tool in tools]
                body["tool_choice"] = "auto"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            excerpt = error.read(2_000).decode("utf-8", errors="replace")
            raise RuntimeError(f"API request failed (HTTP {error.code}): {excerpt}") from error
        if self.api_style == "responses":
            return self._parse_responses(payload)
        return self._parse_chat(payload)

    @staticmethod
    def _chat_tool(tool: AgentTool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            },
        }

    @staticmethod
    def _responses_tool(tool: AgentTool) -> dict[str, Any]:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters),
        }

    @staticmethod
    def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "tool":
                output.append({
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message.get("content", ""),
                })
                continue
            if role == "assistant" and message.get("tool_calls"):
                content = message.get("content")
                if content:
                    output.append({"role": "assistant", "content": content})
                for call in message["tool_calls"]:
                    function = call["function"]
                    output.append({
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": function["name"],
                        "arguments": function["arguments"],
                    })
                continue
            output.append({"role": role, "content": message.get("content", "")})
        return output

    @staticmethod
    def _arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw or "{}")
        except (json.JSONDecodeError, TypeError):
            return {"__invalid_json__": str(raw)}
        return parsed if isinstance(parsed, dict) else {"__invalid_json__": str(raw)}

    @classmethod
    def _parse_chat(cls, payload: dict[str, Any]) -> AssistantMessage:
        try:
            choice = payload["choices"][0]
            raw_message = choice["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"incompatible Chat Completions response: {payload!r}") from error
        content = []
        text = raw_message.get("content")
        if isinstance(text, str) and text:
            content.append(TextContent(text))
        for raw_call in raw_message.get("tool_calls") or []:
            function = raw_call.get("function") or {}
            content.append(ToolCall(
                id=str(raw_call.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=cls._arguments(function.get("arguments")),
            ))
        reason = choice.get("finish_reason")
        stop_reason = "length" if reason == "length" else "tool_calls" if content and any(
            isinstance(item, ToolCall) for item in content
        ) else "stop"
        return AssistantMessage(tuple(content), stop_reason=stop_reason)

    @classmethod
    def _parse_responses(cls, payload: dict[str, Any]) -> AssistantMessage:
        content = []
        for item in payload.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        content.append(TextContent(part["text"]))
            elif item.get("type") == "function_call":
                content.append(ToolCall(
                    id=str(item.get("call_id") or item.get("id") or ""),
                    name=str(item.get("name") or ""),
                    arguments=cls._arguments(item.get("arguments")),
                ))
        incomplete = (payload.get("incomplete_details") or {}).get("reason")
        stop_reason = "length" if incomplete else "tool_calls" if any(
            isinstance(item, ToolCall) for item in content
        ) else "stop"
        return AssistantMessage(tuple(content), stop_reason=stop_reason)

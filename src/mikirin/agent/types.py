from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias


StopReason = Literal["stop", "tool_calls", "length", "error", "aborted"]
ExecutionMode = Literal["parallel", "sequential"]


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]


AssistantContent: TypeAlias = TextContent | ToolCall


@dataclass(frozen=True, slots=True)
class UserMessage:
    content: str
    role: Literal["user"] = "user"


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    content: tuple[AssistantContent, ...]
    stop_reason: StopReason = "stop"
    error_message: str | None = None
    role: Literal["assistant"] = "assistant"

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.content if isinstance(part, TextContent))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(part for part in self.content if isinstance(part, ToolCall))


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)
    role: Literal["tool_result"] = "tool_result"


AgentMessage: TypeAlias = UserMessage | AssistantMessage | ToolResultMessage


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    content: str
    details: Mapping[str, Any] = field(default_factory=dict)


ToolUpdateSink: TypeAlias = Callable[[str], Awaitable[None]]


class AgentTool(Protocol):
    name: str
    description: str
    parameters: Mapping[str, Any]
    execution_mode: ExecutionMode

    async def execute(
        self,
        tool_call_id: str,
        arguments: Mapping[str, Any],
        update: ToolUpdateSink,
    ) -> ToolExecutionResult: ...


@dataclass(slots=True)
class AgentContext:
    system_prompt: str
    messages: list[AgentMessage] = field(default_factory=list)
    tools: tuple[AgentTool, ...] = ()


@dataclass(frozen=True, slots=True)
class AssistantStreamEvent:
    type: Literal["start", "update", "done"]
    message: AssistantMessage
    delta: str | None = None


class AgentProvider(Protocol):
    def stream(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: tuple[AgentTool, ...],
    ) -> AsyncIterator[AssistantStreamEvent]: ...


EventType = Literal[
    "agent_start",
    "turn_start",
    "message_start",
    "message_update",
    "message_end",
    "tool_execution_start",
    "tool_execution_update",
    "tool_execution_end",
    "turn_end",
    "agent_end",
]


@dataclass(frozen=True, slots=True)
class AgentEvent:
    type: EventType
    message: AgentMessage | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResultMessage | None = None
    delta: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)


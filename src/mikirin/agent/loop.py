from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from mikirin.agent.convert import convert_to_chat_messages
from mikirin.agent.types import (
    AgentContext,
    AgentEvent,
    AgentMessage,
    AgentProvider,
    AssistantMessage,
    ToolCall,
    ToolExecutionResult,
    ToolResultMessage,
)
from mikirin.agent.validation import validate_tool_arguments
from mikirin.scrub import StreamingContextScrubber, scrub_assistant_message


EventSink = Callable[[AgentEvent], Awaitable[None] | None]
MessageSource = Callable[[], Awaitable[list[AgentMessage]] | list[AgentMessage]]
ContextTransform = Callable[
    [list[AgentMessage]], Awaitable[list[AgentMessage]] | list[AgentMessage]
]
MessageConverter = Callable[[list[AgentMessage]], list[dict[str, Any]]]
BeforeToolCall = Callable[
    [ToolCall], Awaitable[str | None] | str | None
]
AfterToolCall = Callable[
    [ToolCall, ToolExecutionResult, bool],
    Awaitable[tuple[ToolExecutionResult, bool] | None] | tuple[ToolExecutionResult, bool] | None,
]
StopHook = Callable[
    [AssistantMessage, list[ToolResultMessage]], Awaitable[bool] | bool
]


@dataclass(slots=True)
class AgentConfig:
    provider: AgentProvider
    transform_context: ContextTransform | None = None
    convert_to_llm: MessageConverter = convert_to_chat_messages
    before_tool_call: BeforeToolCall | None = None
    after_tool_call: AfterToolCall | None = None
    get_steering_messages: MessageSource | None = None
    get_follow_up_messages: MessageSource | None = None
    should_stop_after_turn: StopHook | None = None
    max_turns: int = 32


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _emit(sink: EventSink | None, event: AgentEvent) -> None:
    if sink is not None:
        await _resolve(sink(event))


async def _messages(source: MessageSource | None) -> list[AgentMessage]:
    if source is None:
        return []
    return list(await _resolve(source()))


async def _stream_assistant(
    context: AgentContext,
    config: AgentConfig,
    emit: EventSink | None,
) -> AssistantMessage:
    messages = list(context.messages)
    if config.transform_context is not None:
        messages = list(await _resolve(config.transform_context(messages)))
    llm_messages = config.convert_to_llm(messages)
    partial_added = False
    final: AssistantMessage | None = None
    scrubber = StreamingContextScrubber()
    async for event in config.provider.stream(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=context.tools,
    ):
        sanitized = scrub_assistant_message(event.message)
        final = sanitized
        if event.type == "start":
            context.messages.append(sanitized)
            partial_added = True
            await _emit(emit, AgentEvent("message_start", message=sanitized))
        elif event.type == "update":
            safe_delta = scrubber.feed(event.delta or "")
            if partial_added:
                context.messages[-1] = sanitized
            else:
                context.messages.append(sanitized)
                partial_added = True
                await _emit(emit, AgentEvent("message_start", message=sanitized))
            await _emit(
                emit,
                AgentEvent("message_update", message=sanitized, delta=safe_delta),
            )
        elif event.type == "done":
            tail = scrubber.finish()
            if partial_added:
                context.messages[-1] = sanitized
            else:
                context.messages.append(sanitized)
                partial_added = True
                await _emit(emit, AgentEvent("message_start", message=sanitized))
            if tail:
                await _emit(emit, AgentEvent("message_update", message=sanitized, delta=tail))
            await _emit(emit, AgentEvent("message_end", message=sanitized))
            return sanitized
    if final is None:
        final = AssistantMessage((), stop_reason="error", error_message="provider returned no events")
    if partial_added:
        context.messages[-1] = final
    else:
        context.messages.append(final)
        await _emit(emit, AgentEvent("message_start", message=final))
    await _emit(emit, AgentEvent("message_end", message=final))
    return final


async def _execute_one(
    context: AgentContext,
    call: ToolCall,
    config: AgentConfig,
    emit: EventSink | None,
) -> ToolResultMessage:
    await _emit(emit, AgentEvent("tool_execution_start", tool_call=call))
    tool = next((candidate for candidate in context.tools if candidate.name == call.name), None)
    if tool is None:
        result = ToolExecutionResult(f"unknown tool: {call.name}")
        is_error = True
    else:
        blocked_reason = None
        if config.before_tool_call is not None:
            blocked_reason = await _resolve(config.before_tool_call(call))
        if blocked_reason:
            result = ToolExecutionResult(str(blocked_reason))
            is_error = True
        else:
            async def update(text: str) -> None:
                await _emit(
                    emit,
                    AgentEvent("tool_execution_update", tool_call=call, delta=text),
                )

            try:
                validate_tool_arguments(tool.parameters, call.arguments)
                result = await tool.execute(call.id, call.arguments, update)
                is_error = False
            except Exception as error:
                result = ToolExecutionResult(f"{type(error).__name__}: {error}")
                is_error = True
        if config.after_tool_call is not None:
            replacement = await _resolve(config.after_tool_call(call, result, is_error))
            if replacement is not None:
                result, is_error = replacement
    message = ToolResultMessage(
        tool_call_id=call.id,
        tool_name=call.name,
        content=result.content,
        details=result.details,
        is_error=is_error,
    )
    await _emit(
        emit,
        AgentEvent(
            "tool_execution_end", tool_call=call, tool_result=message, data={"is_error": is_error}
        ),
    )
    await _emit(emit, AgentEvent("message_start", message=message))
    await _emit(emit, AgentEvent("message_end", message=message))
    return message


async def _execute_calls(
    context: AgentContext,
    assistant: AssistantMessage,
    config: AgentConfig,
    emit: EventSink | None,
) -> list[ToolResultMessage]:
    calls = list(assistant.tool_calls)
    if assistant.stop_reason == "length":
        results = [
            ToolResultMessage(
                call.id,
                call.name,
                "tool call rejected because the model response was truncated; resend it completely",
                is_error=True,
            )
            for call in calls
        ]
        for call, result in zip(calls, results, strict=True):
            await _emit(emit, AgentEvent("tool_execution_start", tool_call=call))
            await _emit(
                emit,
                AgentEvent("tool_execution_end", tool_call=call, tool_result=result, data={"is_error": True}),
            )
            await _emit(emit, AgentEvent("message_start", message=result))
            await _emit(emit, AgentEvent("message_end", message=result))
        return results
    sequential = any(
        tool.execution_mode == "sequential"
        for tool in context.tools
        if any(call.name == tool.name for call in calls)
    )
    if sequential:
        return [await _execute_one(context, call, config, emit) for call in calls]
    return list(await asyncio.gather(*[
        _execute_one(context, call, config, emit) for call in calls
    ]))


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentConfig,
    *,
    emit: EventSink | None = None,
) -> list[AgentMessage]:
    if config.max_turns < 1:
        raise ValueError("max_turns must be positive")
    new_messages = list(prompts)
    current = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],
        tools=context.tools,
    )
    await _emit(emit, AgentEvent("agent_start"))
    await _emit(emit, AgentEvent("turn_start"))
    for prompt in prompts:
        await _emit(emit, AgentEvent("message_start", message=prompt))
        await _emit(emit, AgentEvent("message_end", message=prompt))

    pending = await _messages(config.get_steering_messages)
    first_turn = True
    turn_count = 0
    while True:
        has_tool_calls = True
        while has_tool_calls or pending:
            turn_count += 1
            if turn_count > config.max_turns:
                raise RuntimeError(f"agent exceeded max_turns={config.max_turns}")
            if not first_turn:
                await _emit(emit, AgentEvent("turn_start"))
            first_turn = False
            for message in pending:
                current.messages.append(message)
                new_messages.append(message)
                await _emit(emit, AgentEvent("message_start", message=message))
                await _emit(emit, AgentEvent("message_end", message=message))
            pending = []

            assistant = await _stream_assistant(current, config, emit)
            new_messages.append(assistant)
            if assistant.stop_reason in {"error", "aborted"}:
                await _emit(emit, AgentEvent("turn_end", message=assistant))
                await _emit(emit, AgentEvent("agent_end", data={"messages": new_messages}))
                return new_messages

            has_tool_calls = bool(assistant.tool_calls)
            tool_results: list[ToolResultMessage] = []
            if has_tool_calls:
                tool_results = await _execute_calls(current, assistant, config, emit)
                current.messages.extend(tool_results)
                new_messages.extend(tool_results)
            await _emit(
                emit,
                AgentEvent(
                    "turn_end",
                    message=assistant,
                    data={"tool_results": tool_results},
                ),
            )
            if config.should_stop_after_turn is not None and await _resolve(
                config.should_stop_after_turn(assistant, tool_results)
            ):
                await _emit(emit, AgentEvent("agent_end", data={"messages": new_messages}))
                return new_messages
            pending = await _messages(config.get_steering_messages)

        follow_up = await _messages(config.get_follow_up_messages)
        if follow_up:
            pending = follow_up
            continue
        break

    await _emit(emit, AgentEvent("agent_end", data={"messages": new_messages}))
    return new_messages

import asyncio
import unittest
from collections.abc import AsyncIterator, Mapping
from typing import Any

from mikirin.agent import (
    AgentConfig,
    AgentContext,
    AssistantMessage,
    TextContent,
    ToolCall,
    UserMessage,
    run_agent_loop,
)
from mikirin.agent.types import AssistantStreamEvent, ToolExecutionResult


class FakeProvider:
    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = list(responses)
        self.requests: list[list[dict[str, Any]]] = []

    async def stream(self, *, system_prompt, messages, tools) -> AsyncIterator[AssistantStreamEvent]:
        self.requests.append(messages)
        response = self.responses.pop(0)
        yield AssistantStreamEvent("start", AssistantMessage(()))
        if response.text:
            yield AssistantStreamEvent("update", response, delta=response.text)
        yield AssistantStreamEvent("done", response)


class CountingTool:
    name = "search_anime"
    description = "search"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    execution_mode = "parallel"

    def __init__(self) -> None:
        self.calls = 0

    async def execute(
        self, tool_call_id: str, arguments: Mapping[str, Any], update
    ) -> ToolExecutionResult:
        self.calls += 1
        await update("searching")
        return ToolExecutionResult('{"title":"孤独摇滚！"}')


class AgentLoopTests(unittest.TestCase):
    def test_tool_result_is_fed_back_before_final_answer(self) -> None:
        provider = FakeProvider([
            AssistantMessage(
                (ToolCall("call-1", "search_anime", {"query": "孤独摇滚"}),),
                stop_reason="tool_calls",
            ),
            AssistantMessage((TextContent("找到了。"),)),
        ])
        tool = CountingTool()
        events = []
        messages = asyncio.run(run_agent_loop(
            [UserMessage("查一下孤独摇滚")],
            AgentContext("system", tools=(tool,)),
            AgentConfig(provider=provider),
            emit=events.append,
        ))
        self.assertEqual(tool.calls, 1)
        self.assertEqual([message.role for message in messages], [
            "user", "assistant", "tool_result", "assistant"
        ])
        self.assertEqual(messages[-1].text, "找到了。")
        self.assertIn("tool_execution_update", [event.type for event in events])
        self.assertEqual(provider.requests[1][-1]["role"], "tool")

    def test_truncated_tool_calls_are_rejected_without_execution(self) -> None:
        provider = FakeProvider([
            AssistantMessage(
                (ToolCall("broken", "search_anime", {"query": "x"}),),
                stop_reason="length",
            ),
            AssistantMessage((TextContent("已重发完整请求。"),)),
        ])
        tool = CountingTool()
        messages = asyncio.run(run_agent_loop(
            [UserMessage("test")],
            AgentContext("system", tools=(tool,)),
            AgentConfig(provider=provider),
        ))
        self.assertEqual(tool.calls, 0)
        self.assertTrue(messages[2].is_error)
        self.assertIn("truncated", messages[2].content)

    def test_invalid_tool_arguments_are_reported_without_execution(self) -> None:
        provider = FakeProvider([
            AssistantMessage(
                (ToolCall("invalid", "search_anime", {"unexpected": "x"}),),
                stop_reason="tool_calls",
            ),
            AssistantMessage((TextContent("参数已修正。"),)),
        ])
        tool = CountingTool()
        messages = asyncio.run(run_agent_loop(
            [UserMessage("test")],
            AgentContext("system", tools=(tool,)),
            AgentConfig(provider=provider),
        ))
        self.assertEqual(tool.calls, 0)
        self.assertTrue(messages[2].is_error)
        self.assertIn("missing required argument", messages[2].content)

    def test_steering_and_follow_up_messages_continue_the_outer_loop(self) -> None:
        provider = FakeProvider([
            AssistantMessage((TextContent("first"),)),
            AssistantMessage((TextContent("second"),)),
        ])
        steering = [[UserMessage("steer")], []]
        follow_up = [[UserMessage("follow")], []]

        def next_steering():
            return steering.pop(0) if steering else []

        def next_follow_up():
            return follow_up.pop(0) if follow_up else []

        messages = asyncio.run(run_agent_loop(
            [UserMessage("initial")],
            AgentContext("system"),
            AgentConfig(
                provider=provider,
                get_steering_messages=next_steering,
                get_follow_up_messages=next_follow_up,
            ),
        ))
        self.assertEqual([message.role for message in messages], [
            "user", "user", "assistant", "user", "assistant"
        ])
        self.assertEqual(provider.requests[0][1]["content"], "steer")
        self.assertEqual(provider.requests[1][-1]["content"], "follow")


if __name__ == "__main__":
    unittest.main()

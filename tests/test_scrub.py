import asyncio
import unittest
from collections.abc import AsyncIterator

from mikirin.agent import AgentConfig, AgentContext, AssistantMessage, TextContent, UserMessage, run_agent_loop
from mikirin.agent.types import AssistantStreamEvent
from mikirin.scrub import StreamingContextScrubber, scrub_text


class SplitFenceProvider:
    async def stream(self, *, system_prompt, messages, tools) -> AsyncIterator[AssistantStreamEvent]:
        yield AssistantStreamEvent("start", AssistantMessage(()))
        yield AssistantStreamEvent(
            "update", AssistantMessage((TextContent("<memory-con"),)), delta="<memory-con"
        )
        yield AssistantStreamEvent(
            "update",
            AssistantMessage((TextContent("<memory-context>背景"),)),
            delta="text>背景",
        )
        yield AssistantStreamEvent(
            "done", AssistantMessage((TextContent("<memory-context>背景</memory-context>回答"),))
        )


class ScrubberTests(unittest.TestCase):
    def test_attributes_whitespace_nested_fences_and_every_stream_boundary(self):
        text = ('前< MEMORY-context kind="x"><conversation-summary>喜好'
                '</conversation-summary></ memory-context >[System note: hidden]后')
        self.assertEqual(scrub_text(text), "前喜好后")
        for split in range(len(text) + 1):
            scrubber = StreamingContextScrubber()
            self.assertEqual(scrubber.feed(text[:split]) + scrubber.feed(text[split:])
                             + scrubber.finish(), "前喜好后")
        self.assertEqual(scrub_text("回答<memory-con"), "回答")
        self.assertEqual(scrub_text("回答[System no"), "回答")

    def test_split_fences_and_system_notes_are_removed(self) -> None:
        scrubber = StreamingContextScrubber()
        pieces = [
            scrubber.feed("开头<memory-con"),
            scrubber.feed("text>[System no"),
            scrubber.feed("te: hidden]内容</memory-context>结尾"),
            scrubber.finish(),
        ]
        self.assertEqual("".join(pieces), "开头内容结尾")

    def test_incomplete_forbidden_tag_is_not_flushed(self) -> None:
        self.assertEqual(scrub_text("回答<memory-context"), "回答")

    def test_agent_events_and_final_message_are_sanitized(self) -> None:
        events = []
        messages = asyncio.run(run_agent_loop(
            [UserMessage("test")],
            AgentContext("system"),
            AgentConfig(provider=SplitFenceProvider()),
            emit=events.append,
        ))
        self.assertEqual(messages[-1].text, "背景回答")
        streamed = "".join(event.delta or "" for event in events if event.type == "message_update")
        self.assertNotIn("memory-context", streamed)


if __name__ == "__main__":
    unittest.main()

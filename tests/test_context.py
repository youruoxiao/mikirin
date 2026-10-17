import asyncio
import unittest
from collections.abc import AsyncIterator
from pathlib import Path

from mikirin.agent.types import AssistantMessage, AssistantStreamEvent, TextContent, UserMessage
from mikirin.chat import KnowledgeAgentSession
from mikirin.context import (
    compaction_prompt,
    estimate_text_tokens,
    recent_turn_cutoff,
)


class SequenceProvider:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.requests = []

    async def stream(self, *, system_prompt, messages, tools) -> AsyncIterator[AssistantStreamEvent]:
        self.requests.append((system_prompt, messages, tools))
        message = AssistantMessage((TextContent(self.texts.pop(0)),))
        yield AssistantStreamEvent("start", AssistantMessage(()))
        yield AssistantStreamEvent("done", message)


class ContextTests(unittest.TestCase):
    def test_cjk_token_estimate_and_turn_boundary(self) -> None:
        self.assertGreater(estimate_text_tokens("孤独摇滚"), estimate_text_tokens("abcd"))
        messages = [
            UserMessage("u1"), AssistantMessage((TextContent("a1"),)),
            UserMessage("u2"), AssistantMessage((TextContent("a2"),)),
            UserMessage("u3"), AssistantMessage((TextContent("a3"),)),
        ]
        self.assertEqual(recent_turn_cutoff(messages, 2), 2)

    def test_incremental_prompt_contains_old_summary_and_fences(self) -> None:
        prompt = compaction_prompt([UserMessage("new")], "old")
        self.assertIn("<previous-summary>", prompt)
        self.assertIn("<conversation>", prompt)
        self.assertIn("PRESERVE", prompt)

    def test_session_compacts_before_main_agent_request(self) -> None:
        provider = SequenceProvider(["压缩后的摘要", "最终回答"])
        records = []
        session = KnowledgeAgentSession(
            database=Path("unused.db"),
            provider=provider,
            history=[
                UserMessage("old-user-" + "x" * 80),
                AssistantMessage((TextContent("old-answer-" + "x" * 80),)),
                UserMessage("recent-user"),
                AssistantMessage((TextContent("recent-answer"),)),
            ],
            system_prompt="base",
            context_window_tokens=80,
            compaction_threshold=0.6,
            keep_recent_turns=1,
            compaction_sink=records.append,
            prompt_rebuilder=lambda summary: "rebuilt:" + summary,
        )
        answer = asyncio.run(session.respond_async("next"))
        self.assertEqual(answer, "最终回答")
        self.assertEqual(session.conversation_summary, "压缩后的摘要")
        self.assertEqual(session.compacted_message_count, 2)
        self.assertEqual(records[0].removed_messages, 2)
        self.assertEqual(session.system_prompt, "rebuilt:压缩后的摘要")
        self.assertEqual(len(provider.requests), 2)
        self.assertEqual(provider.requests[0][2], ())


if __name__ == "__main__":
    unittest.main()

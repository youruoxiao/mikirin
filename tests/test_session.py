import tempfile
import unittest
from pathlib import Path

from mikirin.agent.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from mikirin.session import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_append_only_messages_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            session_id = store.create(metadata={"model": "fixture"})
            messages = [
                UserMessage("你好"),
                AssistantMessage(
                    (ToolCall("c1", "search_anime", {"query": "孤独摇滚"}),),
                    stop_reason="tool_calls",
                ),
                ToolResultMessage("c1", "search_anime", '{"hits":[]}'),
                AssistantMessage((TextContent("没找到。"),)),
            ]
            for message in messages:
                store.append_message(session_id, message)
            replayed = store.replay_messages(session_id)
            self.assertEqual(replayed, messages)
            self.assertEqual(len(store.path(session_id).read_text(encoding="utf-8").splitlines()), 5)

    def test_fork_copies_history_with_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SessionStore(Path(directory))
            original = store.create()
            store.append_message(original, UserMessage("原会话"))
            forked = store.fork(original)
            self.assertNotEqual(forked, original)
            self.assertEqual(store.replay_messages(forked), [UserMessage("原会话")])
            copied = store.entries(forked)[1]
            self.assertIn("origin_entry_id", copied["payload"])
            self.assertEqual(store.replay_messages(original), [UserMessage("原会话")])


if __name__ == "__main__":
    unittest.main()

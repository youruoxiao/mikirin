import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mikirin.chat import KnowledgeChatSession, RemoteAPIChatClient, validate_base_url
from mikirin.layout import ProjectLayout
from mikirin.publish import Publisher
from mikirin.search_index import SearchIndexer


class StubClient:
    def __init__(self) -> None:
        self.messages = []

    def complete(self, messages):
        self.messages = messages
        return "收到"


class ChatTests(unittest.TestCase):
    def test_remote_https_endpoint_is_valid(self) -> None:
        self.assertEqual(
            validate_base_url("https://api.openai.com/v1/"),
            "https://api.openai.com/v1",
        )
        with self.assertRaises(ValueError):
            validate_base_url("not-a-url")

    def test_api_key_is_required(self) -> None:
        with self.assertRaises(ValueError):
            RemoteAPIChatClient(
                base_url="https://api.openai.com/v1",
                model="gpt-5.6-terra",
                api_key="",
            )

    def test_responses_output_text_is_parsed(self) -> None:
        payload = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "你好"}],
                }
            ]
        }
        self.assertEqual(RemoteAPIChatClient._responses_text(payload), "你好")

    def test_responses_request_uses_remote_api_without_exposing_key_in_repr(self) -> None:
        response = io.BytesIO(
            json.dumps(
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "完成"}],
                        }
                    ]
                }
            ).encode("utf-8")
        )
        client = RemoteAPIChatClient(
            base_url="https://api.openai.com/v1",
            model="gpt-5.6-terra",
            api_key="secret-test-key",
        )
        self.assertNotIn("secret-test-key", repr(client))
        with patch("urllib.request.urlopen", return_value=response) as mocked:
            self.assertEqual(
                client.complete([{"role": "user", "content": "你好"}]), "完成"
            )
        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(json.loads(request.data)["model"], "gpt-5.6-terra")

    def test_retrieval_is_injected_as_system_background(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Publisher(ProjectLayout(Path(directory))).create_empty_build(
                build_id="chat", code_version="test"
            )
            SearchIndexer.rebuild(database)
            client = StubClient()
            session = KnowledgeChatSession(database=database, client=client)
            self.assertEqual(session.respond("聊聊孤独摇滚"), "收到")
            self.assertIn("本地知识库命中", client.messages[-2]["content"])
            self.assertEqual(client.messages[-1]["role"], "user")


if __name__ == "__main__":
    unittest.main()

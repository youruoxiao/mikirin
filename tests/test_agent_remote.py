import io
import json
import unittest
from unittest.mock import patch

from mikirin.agent.remote import RemoteAgentProvider
from mikirin.agent.tools import KnowledgeSearchTool
from pathlib import Path


class RemoteAgentProviderTests(unittest.TestCase):
    def test_chat_completion_tool_call_is_parsed_and_schema_is_sent(self) -> None:
        payload = {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_anime",
                            "arguments": '{"query":"孤独摇滚"}',
                        },
                    }],
                },
            }]
        }
        provider = RemoteAgentProvider(
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            api_key="secret-test-key",
        )
        self.assertNotIn("secret-test-key", repr(provider))
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as mocked:
            message = provider._complete(
                system_prompt="system",
                messages=[{"role": "user", "content": "hi"}],
                tools=(KnowledgeSearchTool(Path("content.db")),),
            )
        self.assertEqual(message.stop_reason, "tool_calls")
        self.assertEqual(message.tool_calls[0].arguments["query"], "孤独摇滚")
        request_body = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(request_body["tools"][0]["function"]["name"], "search_anime")
        self.assertEqual(request_body["messages"][0]["role"], "system")

    def test_responses_function_call_is_parsed(self) -> None:
        payload = {
            "output": [{
                "type": "function_call",
                "call_id": "call-2",
                "name": "search_anime",
                "arguments": '{"query":"轻音少女"}',
            }]
        }
        message = RemoteAgentProvider._parse_responses(payload)
        self.assertEqual(message.stop_reason, "tool_calls")
        self.assertEqual(message.tool_calls[0].name, "search_anime")


if __name__ == "__main__":
    unittest.main()

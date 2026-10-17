import io
import json
import unittest
from unittest.mock import patch

from mikirin.deepseek import StructuredAPIClient


class StructuredAPIClientTests(unittest.TestCase):
    def test_deepseek_json_mode_request_and_secret_repr(self) -> None:
        response = io.BytesIO(json.dumps({
            "choices": [{"message": {"content": "{\"decisions\":[]}"}}]
        }).encode("utf-8"))
        client = StructuredAPIClient(api_key="secret", max_attempts=1)
        self.assertNotIn("secret", repr(client))
        with patch("urllib.request.urlopen", return_value=response) as mocked:
            payload = client.complete_json([{"role": "user", "content": "return json"}])
        self.assertEqual(client.response_content(payload), '{"decisions":[]}')
        request_body = json.loads(mocked.call_args.args[0].data)
        self.assertEqual(request_body["response_format"], {"type": "json_object"})
        self.assertEqual(request_body["thinking"], {"type": "enabled"})
        self.assertNotIn("secret", request_body)

    def test_siliconflow_qwen_disables_thinking(self) -> None:
        response = io.BytesIO(json.dumps({
            "choices": [{"message": {"content": "{\"decisions\":[]}"}}]
        }).encode("utf-8"))
        client = StructuredAPIClient(
            api_key="secret",
            base_url="https://api.siliconflow.cn/v1",
            model="Qwen/Qwen3-8B",
            provider="siliconflow",
            max_attempts=1,
        )
        with patch("urllib.request.urlopen", return_value=response) as mocked:
            client.complete_json([{"role": "user", "content": "return json"}], max_tokens=512)
        request_body = json.loads(mocked.call_args.args[0].data)
        self.assertFalse(request_body["enable_thinking"])
        self.assertEqual(request_body["max_tokens"], 512)
        self.assertNotIn("max_completion_tokens", request_body)


if __name__ == "__main__":
    unittest.main()

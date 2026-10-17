import unittest
from unittest.mock import Mock, patch

from mikirin.model_config import ModelProfile
from mikirin.structured_role import client_for_profile


class StructuredRoleTests(unittest.TestCase):
    def test_responses_role_uses_responses_transport_and_checks_json(self):
        profile = ModelProfile('custom', 'https://example.test/v1', 'test', 'responses', 'KEY')
        with patch('mikirin.structured_role.RemoteAPIChatClient') as transport:
            client = client_for_profile(profile, 'fake')
            self.assertEqual(transport.call_args.kwargs['api_style'], 'responses')
            transport.return_value.complete.return_value = '{"action":"skip"}'
            result = client.complete_json([])
            self.assertIn('skip', client.response_content(result))
            transport.return_value.complete.return_value = 'not JSON'
            with self.assertRaises(ValueError):
                client.complete_json([])

    def test_chat_role_retains_json_mode_transport(self):
        profile = ModelProfile('custom', 'https://example.test/v1', 'test', 'chat-completions', 'KEY')
        factory = Mock()
        self.assertIs(client_for_profile(profile, 'fake', chat_factory=factory), factory.return_value)
        factory.assert_called_once()

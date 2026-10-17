"""Use the configured API style for JSON-based memory/review roles."""

from __future__ import annotations

import json

from mikirin.chat import RemoteAPIChatClient
from mikirin.deepseek import StructuredAPIClient
from mikirin.model_config import ModelProfile


class ResponsesJSONClient:
    def __init__(self, profile: ModelProfile, api_key: str) -> None:
        self.client = RemoteAPIChatClient(base_url=profile.base_url, model=profile.model,
                                         api_key=api_key, api_style='responses')

    def complete_json(self, messages, **kwargs):
        # The shared Responses transport requests text; downstream schema checks
        # remain mandatory. Never reinterpret a non-JSON reply as an empty result.
        text = self.client.complete([{'role': 'system', 'content': 'Return only a valid JSON object, without Markdown fences.'}, *messages])
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError('structured role requires a JSON object')
        return {'choices': [{'message': {'content': text}}]}

    response_content = staticmethod(StructuredAPIClient.response_content)


def client_for_profile(profile: ModelProfile, api_key: str, *, chat_factory=StructuredAPIClient):
    if profile.api_style == 'responses':
        return ResponsesJSONClient(profile, api_key)
    if profile.api_style != 'chat-completions':
        raise ValueError('unsupported structured role API style')
    return chat_factory(api_key=api_key, base_url=profile.base_url,
                        model=profile.model, provider=profile.provider)

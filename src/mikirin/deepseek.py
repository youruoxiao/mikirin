from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StructuredAPIClient:
    api_key: str = field(repr=False)
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    provider: str = "deepseek"
    timeout_seconds: float = 300.0
    max_attempts: int = 3

    def __post_init__(self) -> None:
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("structured API base URL must be an HTTPS URL")
        self.base_url = self.base_url.rstrip("/")
        self.api_key = self.api_key.strip()
        self.model = self.model.strip()
        if not self.api_key:
            raise ValueError("structured API key is empty")
        if not self.model:
            raise ValueError("DeepSeek model is empty")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        reasoning_effort: str = "high",
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        if self.provider == "deepseek":
            body.update(
                {
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": reasoning_effort,
                    "max_tokens": max_tokens,
                }
            )
        elif self.provider == "siliconflow":
            body.update(
                {
                    "enable_thinking": False,
                    "max_tokens": max_tokens,
                }
            )
        else:
            body["max_completion_tokens"] = max_tokens
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.load(response)
                content = self.response_content(payload)
                if content:
                    return payload
                last_error = RuntimeError("DeepSeek returned empty message.content")
            except urllib.error.HTTPError as error:
                response_text = error.read(2_000).decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"DeepSeek request failed (HTTP {error.code}): {response_text}"
                )
                if error.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_error from error
            except urllib.error.URLError as error:
                last_error = RuntimeError(f"DeepSeek network request failed: {error.reason}")
            if attempt < self.max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        raise last_error

    @staticmethod
    def response_content(payload: dict[str, Any]) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("DeepSeek response does not contain choices[0].message") from error
        if content is None:
            return ""
        if not isinstance(content, str):
            raise RuntimeError("DeepSeek message.content is not a string")
        return content.strip()


# Stable public name retained for the DeepSeek-first configuration.
DeepSeekClient = StructuredAPIClient

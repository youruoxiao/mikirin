from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mikirin.agent.types import AssistantMessage


_FENCE = re.compile(
    r"<\s*/?\s*(?:memory-context|conversation|previous-summary|conversation-summary|watch-progress)"
    r"\b[^>]*>", re.I
)
_SYSTEM_NOTE = re.compile(r"\[System note:[^\]]*\]", re.I)


class StreamingContextScrubber:
    """Strip memory fence labels even when delimiters cross stream chunks."""

    def __init__(self) -> None:
        self._pending = ""
        self._mode: str | None = None

    def feed(self, chunk: str) -> str:
        output: list[str] = []
        index = 0
        while index < len(chunk):
            if self._mode is None:
                next_angle = chunk.find("<", index)
                next_bracket = chunk.find("[", index)
                candidates = [value for value in (next_angle, next_bracket) if value >= 0]
                if not candidates:
                    output.append(chunk[index:])
                    break
                marker = min(candidates)
                output.append(chunk[index:marker])
                self._pending = chunk[marker]
                self._mode = "angle" if chunk[marker] == "<" else "bracket"
                index = marker + 1
                continue

            terminator = ">" if self._mode == "angle" else "]"
            end = chunk.find(terminator, index)
            if end < 0:
                self._pending += chunk[index:]
                break
            self._pending += chunk[index : end + 1]
            if self._mode == "angle":
                if not _FENCE.fullmatch(self._pending):
                    output.append(self._pending)
            elif not _SYSTEM_NOTE.fullmatch(self._pending):
                output.append(self._pending)
            self._pending = ""
            self._mode = None
            index = end + 1
        return "".join(output)

    def finish(self) -> str:
        pending = self._pending
        self._pending = ""
        mode = self._mode
        self._mode = None
        lowered = pending.casefold()
        if mode == "angle":
            name = re.sub(r"^<\s*/?\s*", "", lowered)
            if name and any(tag.startswith(name) or name.startswith(tag) for tag in (
                "memory-context", "conversation", "previous-summary",
                "conversation-summary", "watch-progress",
            )):
                return ""
        if mode == "bracket" and (
            lowered.startswith("[system note:") or "[system note:".startswith(lowered)
        ):
            return ""
        return pending


def scrub_text(text: str) -> str:
    scrubber = StreamingContextScrubber()
    return scrubber.feed(text) + scrubber.finish()


def scrub_assistant_message(message: AssistantMessage) -> AssistantMessage:
    from mikirin.agent.types import AssistantMessage, TextContent

    content = tuple(
        TextContent(scrub_text(part.text)) if isinstance(part, TextContent) else part
        for part in message.content
    )
    return AssistantMessage(content, message.stop_reason, message.error_message)

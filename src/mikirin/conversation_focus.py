"""对话焦点：跟踪当前话题锚定的作品/单集（v4.1 A3.4 契约）。

- 检索命中更新焦点；作品切换清空单集锚点；
- 超过 decay_turns 轮未刷新则置信度衰减，低于阈值清空；
- 全部确定性，无 LLM。
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class ConversationFocus:
    active_subject_id: str | None = None
    active_episode_id: str | None = None
    active_episode_sort: float | None = None
    subject_name: str = ""
    confidence: float = 0.0
    turns_since_refresh: int = 0

    @property
    def is_empty(self) -> bool:
        return self.active_subject_id is None


EMPTY_FOCUS = ConversationFocus()


def refresh_focus(
    focus: ConversationFocus,
    *,
    subject_id: str,
    subject_name: str,
    episode_id: str | None = None,
    episode_sort: float | None = None,
    confidence: float = 0.9,
) -> ConversationFocus:
    """新的检索锚点刷新焦点；作品切换时清空单集。"""
    if subject_id != focus.active_subject_id:
        return ConversationFocus(
            active_subject_id=subject_id,
            active_episode_id=episode_id,
            active_episode_sort=episode_sort,
            subject_name=subject_name,
            confidence=confidence,
            turns_since_refresh=0,
        )
    return replace(
        focus,
        active_episode_id=episode_id if episode_id is not None else focus.active_episode_id,
        active_episode_sort=(
            episode_sort if episode_sort is not None else focus.active_episode_sort
        ),
        subject_name=subject_name or focus.subject_name,
        confidence=min(1.0, max(focus.confidence, confidence)),
        turns_since_refresh=0,
    )


def tick_focus(
    focus: ConversationFocus,
    *,
    decay_turns: int = 10,
    decay_step: float = 0.15,
    clear_below: float = 0.3,
) -> ConversationFocus:
    """每轮未刷新时调用：衰减置信度，低于阈值清空。"""
    if focus.is_empty:
        return focus
    turns = focus.turns_since_refresh + 1
    confidence = focus.confidence - (decay_step if turns > decay_turns else 0.0)
    if confidence < clear_below:
        return EMPTY_FOCUS
    return replace(focus, turns_since_refresh=turns, confidence=confidence)


def resolve_pronoun(
    focus: ConversationFocus, query: str
) -> tuple[str, str | None, float | None]:
    """查询含"那集/这集/该集/这部/那部"等指代时，返回焦点锚点。

    返回 (是否含指代, subject_id, episode_sort)。不含指代或焦点为空时不解析。
    """
    markers = ("那集", "这集", "该集", "那一集", "这一集", "那部", "这部", "该部",
               "里面", "里面", "那里")
    if focus.is_empty or not any(marker in query for marker in markers):
        return ("", None, None)
    episode = (
        focus.active_episode_sort
        if any(m in query for m in ("那集", "这集", "该集", "那一集", "这一集"))
        else None
    )
    return ("pronoun", focus.active_subject_id, episode)

"""作品内章节引用表与查询集数解析（v4.1-2 契约）。

规则：
- 数字集数不得直接等于 episode sort——经引用表映射；
- type=0 正篇才参与数字映射；SP/OVA/特番（type=1+）只按 sort 精确值；
- "上一集/下一集"仅在存在强 episode 锚时解析（调用方负责）；
- 范围表达（前三集/前八话）返回 range，不伪装单集。
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn_number(text: str) -> int | None:
    """简易中文数字解析（一到九十九，覆盖绝大多数番剧集数）。"""
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if "十" in text:
        left, _, right = text.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        if (left and left not in _CN_DIGITS) or (right and right not in _CN_DIGITS):
            return None
        return tens * 10 + ones
    if len(text) == 1 and text in _CN_DIGITS:
        return _CN_DIGITS[text]
    return None


_EP_PATTERNS = [
    # 第8集 / 第8话 / 第八集 / 第八话 / 第 8 集
    re.compile(r"第\s*([0-9]{1,3}|[零一二两三四五六七八九十]{1,3})\s*[集话回]"),
    # EP8 / ep8 / Ep08 / E08
    re.compile(r"(?i)\bep?[.\s]*([0-9]{1,3})\b"),
    # 8集 / 8话（数字直接跟集话，不带"第"，需词边界防误伤）
    re.compile(r"(?<![0-9第])([0-9]{1,2})\s*[集回](?![话集回0-9])"),
]

_RANGE_PATTERN = re.compile(
    r"前\s*([0-9]{1,3}|[零一二两三四五六七八九十]{1,3})\s*[集话回]"
)


@dataclass(frozen=True, slots=True)
class EpisodeRef:
    episode_id: str
    sort: float
    ep_type: int
    display_number: int | None  # 正篇才有：sort 为整数时的集号
    name: str
    name_cn: str


@dataclass(frozen=True, slots=True)
class EpisodeMention:
    """查询中解析出的集数提及。"""

    kind: str                    # 'episode' / 'range_end'
    display_number: int          # 提到的集号（range 时是上限）
    matched_text: str


class EpisodeTable:
    """单作品的章节引用表。"""

    def __init__(self, episodes: list[EpisodeRef], completed: bool) -> None:
        self.episodes = episodes
        self.completed = completed
        self.main_by_number: dict[int, EpisodeRef] = {}
        for ep in episodes:
            if ep.ep_type == 0 and ep.display_number is not None:
                # 同一 display_number 只保留第一个正篇（防 dump 异常重复）
                self.main_by_number.setdefault(ep.display_number, ep)

    @classmethod
    def load(cls, db: sqlite3.Connection, subject_entity_id: str) -> EpisodeTable:
        rows = db.execute(
            """
            SELECT entity_id, sort, type, name, name_cn
            FROM episodes WHERE subject_entity_id = ? ORDER BY sort, type
            """,
            (subject_entity_id,),
        ).fetchall()
        refs = []
        for entity_id, sort, ep_type, name, name_cn in rows:
            display = int(sort) if ep_type == 0 and float(sort).is_integer() else None
            refs.append(EpisodeRef(entity_id, float(sort), int(ep_type), display,
                                   name or "", name_cn or ""))
        # 完结判定：最后正篇 airdate 已过且无更高 sort 预告（简化：平台字段不判）
        return cls(refs, completed=False)

    def resolve_number(self, display_number: int) -> EpisodeRef | None:
        return self.main_by_number.get(display_number)


def parse_episode_mentions(query: str) -> list[EpisodeMention]:
    """从查询文本解析集数提及（不做作品内映射，那是 resolve 的事）。"""
    mentions: list[EpisodeMention] = []
    seen_spans: list[tuple[int, int]] = []
    for match in _RANGE_PATTERN.finditer(query):
        number = _cn_number(match.group(1))
        if number is not None:
            mentions.append(EpisodeMention("range_end", number, match.group(0)))
            seen_spans.append(match.span())
    for pattern in _EP_PATTERNS:
        for match in pattern.finditer(query):
            if any(s <= match.start() < e for s, e in seen_spans):
                continue
            number = _cn_number(match.group(1))
            if number is None or number < 1 or number > 200:
                continue
            mentions.append(EpisodeMention("episode", number, match.group(0)))
            seen_spans.append(match.span())
    return mentions


def episode_table_for(database: Path, subject_entity_id: str) -> EpisodeTable:
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        return EpisodeTable.load(db, subject_entity_id)

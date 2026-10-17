"""单集检索卡确定性生成（v4.1 契约：禁止为补齐字段引入 LLM）。

卡片 = 作品名及别名 + 集数/标题 + 官方剧情简介
      + 确定性来源的角色/staff + 评论高区分度 n-gram + 去重代表评论。

高区分度 n-gram 用 IDF 思路：该集评论里高频、全库评论里低频的词，
才是"天台哭""live 翻车"这类能定位单集的信号词。
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

CARD_SCHEMA_VERSION = "card-v1"

_TOKEN_RE = re.compile(r"[一-鿿]{2,4}|[a-zA-Z]{2,12}")
_STOP = {
    "什么", "怎么", "为什么", "这个", "那个", "就是", "还是", "不是", "没有",
    "感觉", "觉得", "真的", "现在", "这样", "那样", "知道", "看到", "听到",
    "来说", "一下", "这么", "那么", "如果", "因为", "所以", "但是", "而且",
    "已经", "还是", "其实", "大家", "自己", "我们", "你们", "他们",
}


@dataclass(frozen=True, slots=True)
class EpisodeCard:
    episode_id: str
    subject_id: str
    text: str
    comment_count: int


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text) if t not in _STOP]


def build_episode_cards(
    database: Path,
    *,
    min_comments: int = 3,
    max_representative: int = 3,
) -> list[EpisodeCard]:
    """为有内容（简介或评论达标）的单集生成检索卡。"""
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.row_factory = sqlite3.Row

        # 评论分词的全局文档频率（按"集"为文档单位）
        ep_comments: dict[str, list[tuple[str, int | None]]] = {}
        ep_subject: dict[str, str] = {}
        for row in db.execute(
            """
            SELECT c.episode_entity_id, c.subject_entity_id, d.content, c.rating
            FROM community_document c JOIN source_document d ON d.id=c.document_id
            WHERE c.source_kind='bgm_ep_comment' AND c.episode_entity_id IS NOT NULL
            """
        ):
            ep_comments.setdefault(row["episode_entity_id"], []).append(
                (row["content"], row["rating"])
            )
            ep_subject[row["episode_entity_id"]] = row["subject_entity_id"]

        df: Counter[str] = Counter()
        ep_tokens: dict[str, Counter[str]] = {}
        for episode_id, comments in ep_comments.items():
            counts: Counter[str] = Counter()
            for content, _ in comments:
                counts.update(set(_tokens(content)))
            ep_tokens[episode_id] = counts
            for token in counts:
                df[token] += 1
        total_eps = max(len(ep_tokens), 1)

        cards: list[EpisodeCard] = []
        for row in db.execute(
            """
            SELECT ep.entity_id, ep.subject_entity_id, ep.sort, ep.name, ep.name_cn,
                   ep.description, s.name AS sname, s.name_cn AS sname_cn
            FROM episodes ep JOIN subjects s ON s.entity_id=ep.subject_entity_id
            WHERE ep.type = 0 AND s.type = 2
            ORDER BY ep.subject_entity_id, ep.sort
            """
        ):
            episode_id = row["entity_id"]
            comments = ep_comments.get(episode_id, [])
            has_description = bool((row["description"] or "").strip())
            if not has_description and len(comments) < min_comments:
                continue

            # 角色：该作正/配角（确定性来源）
            characters = [
                r[0]
                for r in db.execute(
                    """
                    SELECT e.canonical_name FROM rel_subject_character rel
                    JOIN entity e ON e.id=rel.character_id
                    WHERE rel.subject_id=? AND rel.role_type IN (1,2)
                    ORDER BY rel.role_type, rel.rel_order LIMIT 8
                    """,
                    (row["subject_entity_id"],),
                )
            ]

            # 高区分度 n-gram：集内 tf × 全局 idf，取 top 8
            distinctive: list[str] = []
            if episode_id in ep_tokens:
                scored = sorted(
                    ep_tokens[episode_id].items(),
                    key=lambda kv: -(kv[1] * math.log(total_eps / (1 + df[kv[0]]))),
                )
                distinctive = [token for token, _ in scored[:8]]

            # 代表评论：长度≥40 且去重，按 评分+长度 取前 N 条截断
            representatives: list[str] = []
            seen: set[str] = set()
            ordered = sorted(
                comments,
                key=lambda item: -((item[1] or 0) * 2 + min(len(item[0]), 400)),
            )
            for content, _rating in ordered:
                key = content[:40]
                if len(content) < 40 or key in seen:
                    continue
                seen.add(key)
                representatives.append(content[:150])
                if len(representatives) >= max_representative:
                    break

            sort = row["sort"]
            ep_label = f"第{int(sort)}集" if float(sort).is_integer() else f"第{sort}集"
            parts = [
                f"作品：{row['sname_cn'] or row['sname']}",
                f"章节：{ep_label}",
            ]
            if row["name"] or row["name_cn"]:
                parts.append(f"标题：{row['name_cn'] or row['name']}")
            if characters:
                parts.append("角色：" + "、".join(characters))
            if has_description:
                parts.append("剧情简介：" + (row["description"] or "").strip()[:300])
            if distinctive:
                parts.append("社区高区分词：" + "、".join(distinctive))
            for text in representatives:
                parts.append("代表性评论：" + text)

            cards.append(
                EpisodeCard(
                    episode_id=episode_id,
                    subject_id=row["subject_entity_id"],
                    text="\n".join(parts),
                    comment_count=len(comments),
                )
            )
        return cards

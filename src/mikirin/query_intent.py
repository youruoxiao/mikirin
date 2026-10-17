"""查询意图规则解析（零 LLM）。

规则覆盖六路路由；拿不准一律 general（退回通用混合配额，
不为分类付 LLM 调用——快路径上不能有不确定延迟）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mikirin.episode_ref import EpisodeMention, parse_episode_mentions

# 方面词典（用于路由和后续 aspect 过滤，非情感结论）
ASPECT_TERMS: dict[str, list[str]] = {
    "animation": ["作画", "原画", "打戏", "崩", "帧数", "动作场面"],
    "direction": ["演出", "分镜", "镜头", "构图", "调度", "漫符"],
    "music": ["音乐", "配乐", "编曲", "音效", "OP", "ED", "插曲", "歌"],
    "voice_acting": ["声优", "配音", "棒读", "演技"],
    "pacing": ["节奏", "拖沓", "赶", "展开太快", "流水账"],
    "character": ["人设", "人物塑造", "成长", "动机", "性格"],
    "adaptation": ["改编", "原作", "删减", "原创", "漫画", "小说"],
    "story": ["剧情", "故事", "伏笔", "展开", "结局", "结尾"],
}

_WHY_RE = re.compile(r"为什么|为何|怎么理解|如何评价.*原因|分析")
_RECEPTION_RE = re.compile(
    r"怎么样|如何|好看吗|口碑|评价|大家怎么看|神作|烂不烂|值不值得|推荐"
)
_FACTUAL_RE = re.compile(r"谁|什么时候|哪年|多少|第几|是不是|有没有|叫什么")
_CHANGE_RE = re.compile(r"怎么变的|变化|前后|从.{1,6}到|转变|成长过程")
_EPISODE_REACTION_RE = re.compile(r"怎么看|怎么样|如何|评价|反应|讨论")
_GLOBAL_THEME_RE = re.compile(r"争议|分歧|为什么.*(分裂|两极)|吵")


@dataclass(frozen=True, slots=True)
class QueryIntent:
    kind: str  # episode_reaction/overall_reception/deep_analysis/cross_episode/
    #            factual/global_theme/general
    episode_mentions: tuple[EpisodeMention, ...] = ()
    aspects: tuple[str, ...] = ()
    raw: str = ""


def detect_aspects(query: str) -> tuple[str, ...]:
    hits = []
    for aspect, terms in ASPECT_TERMS.items():
        if any(term in query for term in terms):
            hits.append(aspect)
    return tuple(hits)


def classify_query(query: str) -> QueryIntent:
    mentions = tuple(parse_episode_mentions(query))
    aspects = detect_aspects(query)

    if mentions:
        # 含集数且带评价词 → 单集反应；含变化词 → 跨集
        if _CHANGE_RE.search(query):
            kind = "cross_episode"
        elif _EPISODE_REACTION_RE.search(query):
            kind = "episode_reaction"
        elif _WHY_RE.search(query):
            kind = "deep_analysis"
        else:
            kind = "episode_reaction"
    elif _GLOBAL_THEME_RE.search(query):
        kind = "global_theme"
    elif _WHY_RE.search(query):
        kind = "deep_analysis"
    elif _CHANGE_RE.search(query):
        kind = "cross_episode"
    elif _RECEPTION_RE.search(query):
        kind = "overall_reception"
    elif _FACTUAL_RE.search(query) and not aspects:
        kind = "factual"
    else:
        kind = "general"
    return QueryIntent(kind=kind, episode_mentions=mentions,
                       aspects=aspects, raw=query)


# 各路意图的文档配额（ep_comment / subject_review / subject_comment）
ROUTE_QUOTAS: dict[str, tuple[int, int, int]] = {
    "episode_reaction": (8, 2, 2),
    "overall_reception": (2, 4, 6),
    "deep_analysis": (3, 8, 1),
    "cross_episode": (8, 3, 2),
    "factual": (1, 1, 1),
    "global_theme": (3, 5, 4),
    "general": (4, 4, 4),
}

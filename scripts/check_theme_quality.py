# -*- coding: utf-8 -*-
"""主题质量程序化校验。

两条硬规则：
1. 模板正则：命中即打回（方面词拼盘/空泛分类/本集万能句）；
2. 具体性：主题必须与其引文共享至少一个具体内容词
   （非方面词、非停用词、长度≥2），否则打回。

用法：python scripts/check_theme_quality.py [maps.jsonl 路径]
默认检查 data/staging/discussion_maps/maps.jsonl，
输出 quality_report.json 与 rejected_themes.jsonl。
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

ASPECT_WORDS = {
    "剧情", "角色", "作画", "演出", "音乐", "节奏", "改编", "声优", "人物",
    "叙事", "分镜", "配音", "塑造", "关系", "表现", "推进", "展开", "呈现",
    "细节", "讨论", "评价", "主题", "方面", "观众", "评论", "本集", "处理",
    "story", "character", "visual", "music", "discussion", "episode",
    "narrative", "production", "review", "comments",
}
STOP_WORDS = {
    "什么", "怎么", "为什么", "这个", "那个", "就是", "还是", "没有", "感觉",
    "觉得", "真的", "可以", "大家", "我们", "一个", "一些", "这种", "那种",
    "怎么样", "如何", "是否", "进行", "通过", "以及", "或者", "但是", "因为",
}
TEMPLATE_RES = [
    re.compile(r"^(剧情|角色|作画|演出|音乐|节奏|改编|声优|人物)[与、和].{0,12}(塑造|关系|表现|推进|展开|呈现|细节|处理|评价)$"),
    re.compile(r"^(剧情|角色|作画|演出|音乐|故事|画面|节奏)[与、和](剧情|角色|作画|演出|音乐|故事|画面|节奏)$"),
    re.compile(r"^(story|character|visual|music|episode|narrative)\b.*\b(discussion|review|summary)$", re.I),
    re.compile(r"^episode\b.*\bdiscussion of\b", re.I),
    re.compile(r"^comments\s", re.I),
    re.compile(r"^本集评论主题\d*$"),
    re.compile(r"^(观众|评论)(对|围绕)(第?.{0,4}集|本集).{0,10}(评价|讨论|反应)$"),
    re.compile(r"^.{0,6}(方面|相关讨论)$"),
]
TOKEN_RE = re.compile(r"[一-鿿]{2,4}|[a-zA-Z]{3,12}")
# Scraped page navigation and tracking fragments are not audience discussion,
# even when a model can repeat them exactly in both a label and a quote.
NON_DISCUSSION_RE = re.compile(
    r"https?://|\bwww\.|\b(?:fr|ref|utm_[a-z]+)=[^\s]+",
    re.IGNORECASE,
)
CLIENT_SIGNATURE_RE = re.compile(
    r"\[来自\s*Bangumi(?:\s+M1)?\s+for\s+android\]\s*(?:获取)?",
    re.IGNORECASE,
)
ATTRIBUTION_THEME_RE = re.compile(
    r"^[^“”\"'，。！？；\n]{1,80}\s+说\s*[:：]"
)
USERNAME_FRAGMENT_RE = re.compile(
    r"^[a-z][a-z0-9_-]{2,30}[-_][^\s]{0,8}[？?]$", re.IGNORECASE
)
SHORT_REACTION_RE = re.compile(r"^(?:yippee|lol|www+)[!！。．…~～：:]*$", re.IGNORECASE)


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text)}


def is_template(theme: str) -> bool:
    theme = theme.strip()
    return any(rx.search(theme) for rx in TEMPLATE_RES)


def has_non_discussion_artifact(text: str) -> bool:
    """Whether text is an URL/navigation fragment rather than a comment."""
    return bool(NON_DISCUSSION_RE.search(text) or CLIENT_SIGNATURE_RE.search(text))


def is_low_information_theme(text: str) -> bool:
    """Reject client signatures, usernames, and reaction-only repetitions."""
    value = " ".join(str(text or "").split()).strip()
    if not value:
        return True
    if ATTRIBUTION_THEME_RE.search(value) or USERNAME_FRAGMENT_RE.fullmatch(value):
        return True
    if SHORT_REACTION_RE.fullmatch(value):
        return True
    repeated_parts = [
        part.strip().casefold()
        for part in re.split(r"[：:/／|]+", value)
        if part.strip()
    ]
    if (
        2 <= len(repeated_parts) <= 3
        and len(set(repeated_parts)) == 1
        and SHORT_REACTION_RE.fullmatch(repeated_parts[0])
    ):
        return True
    compact = "".join(ch.lower() for ch in value if ch.isalnum())
    if len(compact) >= 6 and len(set(compact)) <= 2:
        return True
    return False


def specific_tokens(theme: str, quote_texts: list[str]) -> set[str]:
    theme_tokens = tokens(theme) - ASPECT_WORDS - STOP_WORDS
    theme_tokens = {t for t in theme_tokens if len(t) >= 2}
    quote_tokens = set()
    for q in quote_texts:
        quote_tokens |= tokens(q)
    return theme_tokens & quote_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "maps", nargs="?", type=Path,
        default=Path("data/staging/discussion_maps/maps.jsonl"),
    )
    path = parser.parse_args().maps
    rejected, stats = [], Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = json.loads(line)
        for theme in m.get("themes", []):
            stats["total"] += 1
            theme_text = (theme.get("theme") or "").strip()
            quotes = [q.get("quote", "") for q in theme.get("quotes", [])]
            reasons = []
            if "?" in theme_text:
                reasons.append("encoding_corruption")
            if has_non_discussion_artifact(theme_text) or any(
                has_non_discussion_artifact(q) for q in quotes
            ):
                reasons.append("non_discussion_artifact")
            if is_template(theme_text):
                reasons.append("template_theme")
            if is_low_information_theme(theme_text):
                reasons.append("low_information_theme")
            if not specific_tokens(theme_text, quotes):
                reasons.append("no_specific_token")
            if reasons:
                stats["rejected"] += 1
                for r in reasons:
                    stats[f"reason_{r}"] += 1
                rejected.append({
                    "episode_entity_id": m.get("episode_entity_id"),
                    "theme": theme_text, "reasons": reasons,
                })
    total = max(1, stats["total"])
    report = {
        "file": str(path),
        "themes_total": stats["total"],
        "themes_passed": stats["total"] - stats["rejected"],
        "themes_rejected": stats["rejected"],
        "pass_rate": round((stats["total"] - stats["rejected"]) / total, 4),
        "reject_reasons": {
            k.replace("reason_", ""): v for k, v in stats.items() if k.startswith("reason_")
        },
    }
    out_dir = path.parent
    (out_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "rejected_themes.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rejected), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

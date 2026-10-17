#!/usr/bin/env python3
"""Generate episode discussion maps without an LLM.

The extractor is deliberately conservative and extractive:

* split comments into literal, contiguous clauses;
* find clauses from different comments that repeat a concrete phrase;
* rank and de-duplicate those evidence pairs;
* use a concise literal clause as the theme label.

Because labels and quotes come from the source text, the generator cannot
hallucinate episode facts and quote validation is deterministic.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .check_theme_quality import (
        CLIENT_SIGNATURE_RE,
        is_low_information_theme,
        specific_tokens,
    )
else:  # Direct invocation: python scripts/extract_discussion_maps.py.
    from check_theme_quality import (
        CLIENT_SIGNATURE_RE,
        is_low_information_theme,
        specific_tokens,
    )


DEFAULT_BASE = Path("data/staging/discussion_maps/theme-v2-full-20260801")
DEFAULT_OUTPUTS = Path("data/staging/discussion_maps/theme-v2-extractive-20260801/outputs")

CLAUSE_RE = re.compile(r"[^。！？!?；;\n\r]+[。！？!?；;]?")
LATIN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_.+-]{1,31}")
CJK_RUN_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff]{2,}")
ATTRIBUTION_RE = re.compile(
    r"^[^“”\"'，。！？；\n]{1,120}\s+说\s*[:：]\s*"
)
URL_RE = re.compile(r"https?://|\bwww\.|\b(?:fr|ref|utm_[a-z]+)=", re.I)
VOCAB_RE = re.compile(r"^(?:vocab|misc|ps|p\.s\.)\s*[:：]", re.I)
ONLY_NOISE_RE = re.compile(r"^[\W_\d]+$", re.UNICODE)
SCRAPE_TAIL_RE = re.compile(r"(?:\s*error\s+object\s+null\s*)+$", re.I)
TRUNCATED_REPLY_RE = re.compile(r"\.{3,}\s*说\s*[:：]\s*$")
HTML_ENTITY_RE = re.compile(r"&(?:quot|amp|lt|gt|nbsp);", re.I)
META_RE = re.compile(
    r"发太多图|污染讨论楼|图包起来|这一楼里提问|有没有什么作品也是|"
    r"纯好奇|收藏起来慢慢看|字幕下载|点击查看|回复可见",
    re.I,
)

# Exact n-grams that recur in almost any anime discussion.  Longer phrases
# containing these words remain eligible (for example, 动画制作流程).
GENERIC_TERMS = {
    "这个", "那个", "这部", "这集", "本集", "一集", "第一", "第二",
    "感觉", "觉得", "真的", "还是", "就是", "没有", "可以", "一个",
    "一些", "什么", "怎么", "为什么", "不错", "好看", "喜欢", "期待",
    "动画", "动漫", "剧情", "故事", "角色", "人物", "作画", "演出",
    "画面", "音乐", "节奏", "观众", "评论", "主题", "讨论", "表现",
    "处理", "展开", "方面", "进行", "时候", "里面", "这里", "出来",
    "但是", "而且", "因为", "所以", "如果", "不过", "已经", "还有",
    "看到", "看完", "episode", "story", "character", "animation",
}
GENERIC_FRAGMENTS = (
    "我觉得", "感觉", "的时候", "第一遍", "一开始", "看完", "看到",
    "这个", "那个", "还是", "真的", "就是", "可以", "没有", "已经",
)

LEADING_FILLERS = (
    "我觉得", "感觉", "个人觉得", "只能说", "说实话", "不得不说",
    "果然", "真的", "总之", "话说", "不过", "但是", "另外",
)

ASPECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "adaptation": ("原作", "改编", "漫画", "小说", "删减", "原创", "魔改"),
    "voice_acting": ("声优", "配音", "棒读", "声线", "演技", "cv"),
    "music": ("配乐", "音乐", "音效", "插入曲", "主题曲", "op", "ed", "bgm", "唱"),
    "direction": ("分镜", "镜头", "演出", "构图", "转场", "调度", "蒙太奇"),
    "animation": ("作画", "原画", "动作", "打戏", "摄影", "美术", "3d", "cg", "帧"),
    "pacing": ("节奏", "信息量", "拖沓", "仓促", "进度", "暂停", "快进"),
    "character": ("角色", "人物", "男主", "女主", "主角", "成长", "塑造", "关系"),
    "story": ("剧情", "故事", "伏笔", "结局", "展开", "反转", "设定", "时间线"),
    "emotion": ("看哭", "泪目", "感动", "破防", "笑死", "笑哭", "难受", "心疼"),
}


@dataclass(frozen=True)
class Clause:
    comment_id: str
    text: str
    order: int
    tokens: frozenset[str]
    quality: float


@dataclass(frozen=True)
class Candidate:
    first: Clause
    second: Clause
    anchor: str
    shared: frozenset[str]
    support: int
    score: float


def normalize_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    # A few frequent forms materially improve exact-phrase recall while the
    # emitted label and quotes still use untouched source strings.
    replacements = {
        "一分钟": "1分钟", "两分钟": "2分钟", "二分钟": "2分钟",
        "三分钟": "3分钟", "四分钟": "4分钟", "五分钟": "5分钟",
        "六分钟": "6分钟", "七分钟": "7分钟", "八分钟": "8分钟",
        "九分钟": "9分钟", "十分钟": "10分钟",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def useful_token(token: str) -> bool:
    token = token.strip().lower()
    if len(token) < 2 or token in GENERIC_TERMS:
        return False
    if ONLY_NOISE_RE.fullmatch(token):
        return False
    if len(token) <= 8 and any(fragment in token for fragment in GENERIC_FRAGMENTS):
        return False
    return True


def phrase_tokens(text: str) -> frozenset[str]:
    """Return matchable literal-ish terms; longer repeated n-grams rank best."""
    normalized = normalize_match(text)
    found: set[str] = set()
    for word in LATIN_RE.findall(normalized):
        if useful_token(word):
            found.add(word)
    for run in CJK_RUN_RE.findall(normalized):
        # Six characters are enough to be highly episode-specific while
        # keeping the inverted index bounded on long comments.
        for size in range(2, min(6, len(run)) + 1):
            for offset in range(len(run) - size + 1):
                token = run[offset : offset + size]
                if useful_token(token):
                    found.add(token)
    return frozenset(found)


def clause_quality(text: str) -> float:
    content_chars = sum(ch.isalnum() or "\u3400" <= ch <= "\u9fff" for ch in text)
    score = min(content_chars, 80) / 20
    if 12 <= len(text) <= 90:
        score += 2.0
    elif len(text) > 150:
        score -= 1.5
    if any(mark in text for mark in ("因为", "所以", "但是", "相比", "而不是", "却", "导致", "不如")):
        score += 0.8
    if any(mark in text for mark in ("？", "?", "是否", "为什么")):
        score += 0.4
    if URL_RE.search(text) or VOCAB_RE.search(text):
        score -= 10
    return score


def split_comment(comment_id: str, content: str, start_order: int) -> list[Clause]:
    clauses: list[Clause] = []
    order = start_order
    for match in CLAUSE_RE.finditer(content):
        raw = SCRAPE_TAIL_RE.sub("", match.group(0)).strip()
        raw = CLIENT_SIGNATURE_RE.sub("", raw).strip()
        entity = HTML_ENTITY_RE.search(raw)
        if entity:
            raw = raw[: entity.start()].strip()
        if (
            not raw
            or len(raw) < 6
            or URL_RE.search(raw)
            or VOCAB_RE.search(raw)
            or META_RE.search(raw)
            or TRUNCATED_REPLY_RE.search(raw)
        ):
            continue
        if ONLY_NOISE_RE.fullmatch(raw):
            continue
        # Very long runs are split again at commas.  Every emitted piece is
        # still a contiguous source substring.
        pieces = [raw]
        if len(raw) > 180:
            pieces = [piece.strip() for piece in re.split(r"(?<=[，,：:])", raw)]
        for piece in pieces:
            if not (6 <= len(piece) <= 220) or URL_RE.search(piece):
                continue
            tokens = phrase_tokens(piece)
            if not tokens:
                continue
            clauses.append(Clause(comment_id, piece, order, tokens, clause_quality(piece)))
            order += 1
    return clauses


def build_clauses(comments: list[dict]) -> list[Clause]:
    clauses: list[Clause] = []
    for comment in comments:
        comment_id = str(comment.get("comment_id") or "")
        content = str(comment.get("content") or "")
        if not comment_id or not content:
            continue
        clauses.extend(split_comment(comment_id, content, len(clauses)))
    return clauses


def maximal_shared(tokens: set[str]) -> list[str]:
    """Drop n-grams contained in a longer shared term."""
    ranked = sorted(tokens, key=lambda value: (-len(value), value))
    kept: list[str] = []
    for token in ranked:
        if any(token in other for other in kept):
            continue
        kept.append(token)
    return kept


def echo_text(text: str) -> str:
    text = ATTRIBUTION_RE.sub("", text)
    text = normalize_match(text)
    return "".join(ch for ch in text if ch.isalnum() or "\u3400" <= ch <= "\u9fff")


def is_reply_echo(first: str, second: str) -> bool:
    """Detect copied parent text in nested web replies, not independent support."""
    left, right = echo_text(first), echo_text(second)
    if min(len(left), len(right)) < 10:
        return False
    shorter, longer = sorted((left, right), key=len)
    if shorter in longer:
        return True
    left_grams = {left[i : i + 4] for i in range(max(0, len(left) - 3))}
    right_grams = {right[i : i + 4] for i in range(max(0, len(right) - 3))}
    if not left_grams or not right_grams:
        return False
    return len(left_grams & right_grams) / min(len(left_grams), len(right_grams)) >= 0.78


def candidate_pairs(clauses: list[Clause]) -> list[Candidate]:
    postings: dict[str, list[int]] = defaultdict(list)
    for index, clause in enumerate(clauses):
        for token in clause.tokens:
            postings[token].append(index)

    support_by_token: dict[str, int] = {}
    eligible: list[str] = []
    for token, indexes in postings.items():
        support = len({clauses[index].comment_id for index in indexes})
        # Phrases repeated everywhere are normally generic boilerplate.
        if 2 <= support <= 12:
            support_by_token[token] = support
            eligible.append(token)
    eligible.sort(key=lambda token: (-len(token), support_by_token[token], token))
    eligible = eligible[:1200]

    best_by_pair: dict[tuple[int, int], Candidate] = {}
    for anchor in eligible:
        indexes = sorted(
            set(postings[anchor]), key=lambda index: clauses[index].quality, reverse=True
        )[:12]
        for left_pos, left in enumerate(indexes):
            for right in indexes[left_pos + 1 :]:
                first, second = clauses[left], clauses[right]
                if first.comment_id == second.comment_id:
                    continue
                if is_reply_echo(first.text, second.text):
                    continue
                pair_key = (min(left, right), max(left, right))
                shared = set(first.tokens & second.tokens)
                specific = maximal_shared({token for token in shared if token in support_by_token})
                if not specific:
                    continue
                if len(specific[0]) < 3 and support_by_token[specific[0]] < 3:
                    continue
                longest = len(specific[0])
                weighted_overlap = sum(min(len(token), 6) for token in specific[:5])
                support = support_by_token[anchor]
                score = (
                    longest * 2.4
                    + weighted_overlap * 0.45
                    + first.quality
                    + second.quality
                    + math.log2(support + 1)
                )
                candidate = Candidate(
                    first=first,
                    second=second,
                    anchor=anchor,
                    shared=frozenset(specific),
                    support=support,
                    score=score,
                )
                previous = best_by_pair.get(pair_key)
                if previous is None or candidate.score > previous.score:
                    best_by_pair[pair_key] = candidate
    return sorted(best_by_pair.values(), key=lambda item: item.score, reverse=True)


def label_from_clause(text: str, other: str, anchors: frozenset[str]) -> str:
    label = ATTRIBUTION_RE.sub("", text).strip(" \t\r\n-—·•「」『』\"'")
    for prefix in LEADING_FILLERS:
        if label.startswith(prefix) and len(label) > len(prefix) + 8:
            label = label[len(prefix) :].lstrip("，,：: ")
            break
    label = label.replace("?", "？")
    if len(label) > 58:
        cut = max(label.rfind(mark, 24, 59) for mark in "，,：:）)！。；;")
        label = label[: cut + 1 if cut >= 24 else 58].rstrip("，,：:；; ")
    if len(label) < 10:
        supplement = ATTRIBUTION_RE.sub("", other).strip()
        supplement = supplement.replace("?", "？")
        if echo_text(label) != echo_text(supplement):
            label = f"{label}：{supplement[: max(0, 52 - len(label))]}".rstrip("，,：:；; ")
    return " ".join(label.split())


def ensure_auditable_label(label: str, quotes: list[str]) -> str:
    """Keep one validator-visible literal token after shortening a clause.

    The quality validator chunks uninterrupted Chinese into groups of 2--4
    characters.  Removing a discourse prefix can change that chunk alignment
    even though the entire label is copied from a quote.  In that rare case,
    append one literal source token explicitly.
    """
    if specific_tokens(label, quotes):
        return label
    source_tokens: set[str] = set()
    for quote in quotes:
        source_tokens |= specific_tokens(quote, quotes)
    if not source_tokens:
        return label
    anchor = max(source_tokens, key=lambda token: (len(token), token))
    return f"{label}（关键词：{anchor}）"


def infer_aspect(text: str) -> str:
    lowered = normalize_match(text)
    scores = Counter()
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                scores[aspect] += 2 if len(keyword) >= 3 else 1
    if not scores:
        return "other"
    return max(scores, key=lambda aspect: (scores[aspect], aspect != "emotion"))


def label_clause_score(clause: Clause, anchors: frozenset[str]) -> float:
    normalized = normalize_match(clause.text)
    shared_hits = sum(1 for token in anchors if token in normalized)
    latin_or_number = len(re.findall(r"[a-z]{2,}|\d+", normalized))
    contrast = sum(
        marker in clause.text
        for marker in ("但是", "而不是", "相比", "导致", "所以", "却", "是否", "问题")
    )
    return min(len(clause.text), 160) / 24 + shared_hits * 0.35 + latin_or_number * 0.3 + contrast


def overlap_ratio(left: frozenset[str], right: frozenset[str]) -> float:
    left_long = {token for token in left if len(token) >= 3}
    right_long = {token for token in right if len(token) >= 3}
    if not left_long or not right_long:
        return 0.0
    return len(left_long & right_long) / min(len(left_long), len(right_long))


def select_candidates(candidates: list[Candidate], maximum: int) -> list[Candidate]:
    selected: list[Candidate] = []
    used_pairs: set[frozenset[str]] = set()
    for candidate in candidates:
        pair = frozenset((candidate.first.comment_id, candidate.second.comment_id))
        if pair in used_pairs:
            continue
        if any(overlap_ratio(candidate.shared, old.shared) >= 0.55 for old in selected):
            continue
        if any(
            candidate.anchor in old.anchor or old.anchor in candidate.anchor
            for old in selected
            if min(len(candidate.anchor), len(old.anchor)) >= 3
        ):
            continue
        selected.append(candidate)
        used_pairs.add(pair)
        if len(selected) >= maximum:
            break
    return selected


def fallback_candidate(clauses: list[Clause]) -> Candidate | None:
    """Return one literal observation when no cross-comment phrase survives."""
    best_related: Candidate | None = None
    for left, first in enumerate(clauses):
        for second in clauses[left + 1 :]:
            if first.comment_id != second.comment_id or first.text == second.text:
                continue
            shared = maximal_shared(set(first.tokens & second.tokens))
            if not shared or len(shared[0]) < 3:
                continue
            score = len(shared[0]) * 2 + first.quality + second.quality
            candidate = Candidate(
                first, second, shared[0], frozenset(shared), 1, score
            )
            if best_related is None or candidate.score > best_related.score:
                best_related = candidate
    if best_related is not None:
        return best_related

    ranked = sorted(clauses, key=lambda clause: clause.quality, reverse=True)
    if not ranked:
        return None
    first = ranked[0]
    second = next((row for row in ranked[1:] if row.comment_id != first.comment_id), None)
    if second is None:
        second = next((row for row in ranked[1:] if row.text != first.text), first)
    shared = first.tokens & second.tokens
    return Candidate(first, second, "", frozenset(shared), 1, 0.0)


def render_candidate(candidate: Candidate) -> dict | None:
    preferred, other = sorted(
        (candidate.first, candidate.second),
        key=lambda row: label_clause_score(row, candidate.shared),
        reverse=True,
    )
    label = label_from_clause(preferred.text, other.text, candidate.shared)
    quotes = [candidate.first.text, candidate.second.text]
    label = ensure_auditable_label(label, quotes)
    if not specific_tokens(label, quotes) or is_low_information_theme(label):
        return None
    return {
        "theme": label,
        "aspect": infer_aspect(f"{label}\n{candidate.first.text}\n{candidate.second.text}"),
        "quotes": [
            {"comment_id": candidate.first.comment_id, "quote": candidate.first.text},
            {"comment_id": candidate.second.comment_id, "quote": candidate.second.text},
        ],
    }


def extract_map(payload: dict, maximum: int = 5) -> dict:
    clauses = build_clauses(payload.get("comments") or [])
    # Keep only the strongest candidates.  Invalid/low-information candidates
    # reduce the final count instead of causing weaker material to fill quota.
    selected = select_candidates(candidate_pairs(clauses), maximum)
    fallback = fallback_candidate(clauses)

    themes = []
    emotion_used = False
    seen_labels: set[str] = set()
    for candidate in selected:
        rendered = render_candidate(candidate)
        if rendered is None:
            continue
        label = rendered["theme"]
        if not label or label in seen_labels:
            continue
        aspect = rendered["aspect"]
        if aspect == "emotion":
            if emotion_used:
                aspect = "other"
            emotion_used = True
        rendered["aspect"] = aspect
        themes.append(rendered)
        seen_labels.add(label)
        if len(themes) >= maximum:
            break
    if not themes and fallback is not None:
        rendered = render_candidate(fallback)
        if rendered is not None:
            themes.append(rendered)
    return {
        "episode_entity_id": payload.get("episode_entity_id"),
        "themes": themes,
        "generation": "local-extractive-v1",
    }


def parse_batch_names(values: list[str]) -> set[str]:
    return {
        value if value.startswith("batch-") else f"batch-{value.zfill(4)}.jsonl"
        for value in values
    }


def input_files(base: Path, batches: list[str]) -> list[Path]:
    requested = parse_batch_names(batches)
    if not requested:
        return sorted((base / "inputs").glob("*.json"))
    files: list[Path] = []
    for batch in sorted((base / "batches").glob("batch-*.jsonl")):
        if batch.name not in requested:
            continue
        for line in batch.read_text(encoding="utf-8").splitlines():
            if line.strip():
                files.append(base / "inputs" / json.loads(line)["file"])
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--batch", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--maximum-themes", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--overwrite-extractive",
        action="store_true",
        help="overwrite only existing local-extractive-v1 files; preserve Luna seeds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = input_files(args.base, args.batch)
    if args.limit is not None:
        files = files[: args.limit]
    args.outputs.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    for index, source in enumerate(files, 1):
        destination = args.outputs / source.name
        if destination.exists() and not args.overwrite:
            if args.overwrite_extractive:
                try:
                    existing = json.loads(destination.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
                if existing.get("generation") != "local-extractive-v1":
                    stats["skipped"] += 1
                    continue
            else:
                stats["skipped"] += 1
                continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        result = extract_map(payload, maximum=args.maximum_themes)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)
        stats["written"] += 1
        stats["themes"] += len(result["themes"])
        if not result["themes"]:
            stats["empty"] += 1
        if index % 100 == 0:
            print(f"processed={index}/{len(files)} written={stats['written']}", flush=True)
    print(json.dumps({"files": len(files), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

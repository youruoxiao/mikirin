"""Deterministic, tokenizer-free candidate generation for A4.

The module deliberately works on Unicode code points rather than words.  It is
small enough to use in an offline build and keeps source offsets so callers
can attach evidence to the original document.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Iterator


def normalize_for_matching(text: str) -> str:
    """Return the stable text form used for candidate and alias keys."""

    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).casefold().strip()


@dataclass(frozen=True, slots=True)
class CandidateSpan:
    """One candidate occurrence and its source span.

    ``start`` and ``end`` are Python string offsets in the original input;
    ``end`` is exclusive.  ``source`` is one of ``char_ngram``,
    ``latin_kana_digit`` and ``syntax_slot``.
    """

    text: str
    start: int
    end: int
    source: str
    slot_names: tuple[str, ...] = ()

    @property
    def candidate(self) -> str:
        """Compatibility alias for callers that call the value a candidate."""

        return self.text

    @property
    def kind(self) -> str:
        return self.source

    @property
    def span(self) -> tuple[int, int]:
        return self.start, self.end

    @property
    def normalized(self) -> str:
        return normalize_for_matching(self.text)


@dataclass(frozen=True, slots=True)
class SlotPattern:
    """An auditable syntax slot rule."""

    name: str
    expression: str
    description: str

    @property
    def pattern(self) -> str:
        return self.expression


@dataclass(frozen=True, slots=True)
class SlotHit:
    name: str
    candidate: str
    start: int
    end: int
    matched_text: str

    @property
    def slot_name(self) -> str:
        return self.name


def _non_space_runs(text: str) -> Iterator[tuple[int, int]]:
    start: int | None = None
    for index, char in enumerate(text):
        if char.isspace():
            if start is not None:
                yield start, index
                start = None
        elif start is None:
            start = index
    if start is not None:
        yield start, len(text)


def generate_char_ngrams(
    text: str, min_n: int = 1, max_n: int = 5
) -> tuple[CandidateSpan, ...]:
    """Generate Unicode character n-grams without crossing whitespace."""

    if not text:
        return ()
    if min_n < 1 or max_n < min_n:
        raise ValueError("expected 1 <= min_n <= max_n")
    spans: list[CandidateSpan] = []
    for run_start, run_end in _non_space_runs(text):
        for start in range(run_start, run_end):
            for size in range(min_n, max_n + 1):
                end = start + size
                if end > run_end:
                    break
                spans.append(CandidateSpan(text[start:end], start, end, "char_ngram"))
    return tuple(spans)


def _is_latin(char: str) -> bool:
    return "LATIN" in unicodedata.name(char, "")


def _is_kana(char: str) -> bool:
    name = unicodedata.name(char, "")
    return "HIRAGANA" in name or "KATAKANA" in name or char in "ーゝゞヽヾ"


def _is_digit(char: str) -> bool:
    return unicodedata.category(char) == "Nd"


def _is_run_base(char: str) -> bool:
    return _is_latin(char) or _is_kana(char) or _is_digit(char)


def _is_run_joiner(char: str) -> bool:
    return char in "-_+.&'’"


def generate_latin_kana_digit_runs(text: str) -> tuple[CandidateSpan, ...]:
    """Extract contiguous Latin/kana/decimal-digit strings.

    A small set of internal joiners is accepted (for example ``K-ON`` and
    ``v2.0``), but a joiner never starts or ends a run.  Combining marks stay
    with the preceding Latin or kana character.
    """

    if not text:
        return ()
    spans: list[CandidateSpan] = []
    index = 0
    while index < len(text):
        if not _is_run_base(text[index]):
            index += 1
            continue
        start = index
        index += 1
        while index < len(text):
            char = text[index]
            if _is_run_base(char) or unicodedata.category(char).startswith("M"):
                index += 1
                continue
            if _is_run_joiner(char):
                next_index = index + 1
                if next_index < len(text) and _is_run_base(text[next_index]):
                    index += 1
                    continue
            break
        end = index
        while end > start and _is_run_joiner(text[end - 1]):
            end -= 1
        if end > start:
            spans.append(CandidateSpan(text[start:end], start, end, "latin_kana_digit"))
        index = max(index, end + 1 if end == start else index)
    return tuple(spans)


# The capture is deliberately bounded.  Slot patterns are high precision
# evidence; a caller can still obtain longer strings from n-grams.
_TOKEN = (
    r"(?P<candidate>[\w\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff"
    r"\uac00-\ud7af][\w\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff"
    r"\uac00-\ud7af ._+#&'’\-]{0,31}?)"
)
_END = r"(?=$|[\s,，。！？!?；;:：、/／「」『』（）()【】\[\]<>《》をにがはのもでへとや])"


# Keep this tuple explicit.  It is the auditable A4 rule inventory.
SLOT_PATTERNS: tuple[SlotPattern, ...] = (
    SlotPattern("cn_watch", rf"看{_TOKEN}{_END}", "看__"),
    SlotPattern("cn_follow", rf"追{_TOKEN}{_END}", "追__"),
    SlotPattern("cn_recommend", rf"安利{_TOKEN}{_END}", "安利__"),
    SlotPattern("cn_like", rf"喜欢{_TOKEN}{_END}", "喜欢__"),
    SlotPattern("cn_discuss", rf"讨论{_TOKEN}{_END}", "讨论__"),
    SlotPattern("cn_praise", rf"吹爆{_TOKEN}{_END}", "吹爆__"),
    SlotPattern("cn_work_prefix", rf"这部{_TOKEN}{_END}", "这部__"),
    SlotPattern("cn_fan_suffix", rf"{_TOKEN}厨{_END}", "__厨"),
    SlotPattern("cn_party_suffix", rf"{_TOKEN}党{_END}", "__党"),
    SlotPattern("cn_fan_suffix_2", rf"{_TOKEN}粉{_END}", "__粉"),
    SlotPattern("cn_hater_suffix", rf"{_TOKEN}黑{_END}", "__黑"),
    SlotPattern("cn_godlike_suffix", rf"{_TOKEN}神作{_END}", "__神作"),
    SlotPattern("cn_second_season", rf"{_TOKEN}(?:的)?第二季{_END}", "__第二季"),
    SlotPattern("cn_third_season", rf"{_TOKEN}(?:的)?第三季{_END}", "__第三季"),
    SlotPattern("cn_movie", rf"{_TOKEN}剧场版{_END}", "__剧场版"),
    SlotPattern("cn_arc", rf"{_TOKEN}篇{_END}", "__篇"),
    SlotPattern("cn_route", rf"{_TOKEN}线{_END}", "__线"),
    SlotPattern("cn_episode", rf"{_TOKEN}第[一二三四五六七八九十百0-9]+[集话]{_END}", "__第N集/话"),
    SlotPattern("cn_sequel", rf"{_TOKEN}续作{_END}", "__续作"),
    SlotPattern("cn_opinion", rf"{_TOKEN}党争{_END}", "__党争"),
    SlotPattern("jp_watch", rf"(?:見る|観る){_TOKEN}{_END}", "見る/観る__"),
    SlotPattern("jp_follow", rf"追う{_TOKEN}{_END}", "追う__"),
    SlotPattern("jp_push", rf"{_TOKEN}推し{_END}", "__推し"),
    SlotPattern("jp_fan", rf"{_TOKEN}厨{_END}", "__厨"),
    SlotPattern("jp_like", rf"{_TOKEN}好き{_END}", "__好き"),
    SlotPattern("jp_second_season", rf"{_TOKEN}(?:の)?第2期{_END}", "__第2期"),
    SlotPattern("jp_episode", rf"{_TOKEN}第[一二三四五六七八九十0-9]+話{_END}", "__第N話"),
    SlotPattern("jp_movie", rf"{_TOKEN}劇場版{_END}", "__劇場版"),
    SlotPattern("jp_arc", rf"{_TOKEN}編{_END}", "__編"),
    SlotPattern("jp_title_prefix", rf"この{_TOKEN}{_END}", "この__"),
    SlotPattern("en_watch", rf"\bwatch\s+{_TOKEN}{_END}", "watch __"),
    SlotPattern("en_follow", rf"\bfollow\s+{_TOKEN}{_END}", "follow __"),
    SlotPattern("en_recommend", rf"\brecommend\s+{_TOKEN}{_END}", "recommend __"),
    SlotPattern("en_fan", rf"\b{_TOKEN}\s+fans?{_END}", "__ fan/fans"),
    SlotPattern("en_fan_of", rf"\bfans?\s+of\s+{_TOKEN}{_END}", "fans of __"),
    SlotPattern("en_season", rf"\b{_TOKEN}\s+season\s+(?:2|two|second){_END}", "__ season 2"),
    SlotPattern("en_episode", rf"\b{_TOKEN}\s+episode\s+[0-9]+{_END}", "__ episode N"),
    SlotPattern("en_arc", rf"\b{_TOKEN}\s+arc{_END}", "__ arc"),
    SlotPattern("en_movie", rf"\b{_TOKEN}\s+movie{_END}", "__ movie"),
    SlotPattern("en_peak", rf"\b{_TOKEN}\s+is\s+(?:peak|goated|great){_END}", "__ is peak/goated/great"),
    SlotPattern("en_ship", rf"\bship\s+{_TOKEN}{_END}", "ship __"),
)


def find_slot_hits(text: str, patterns: Iterable[SlotPattern] = SLOT_PATTERNS) -> tuple[SlotHit, ...]:
    """Find all syntax-slot captures in stable source/order order."""

    if not text:
        return ()
    hits: list[SlotHit] = []
    for pattern in patterns:
        compiled = re.compile(pattern.expression, re.IGNORECASE | re.UNICODE)
        for match in compiled.finditer(text):
            capture = match.groupdict().get("candidate")
            if not capture:
                continue
            start, end = match.span("candidate")
            hits.append(SlotHit(pattern.name, capture.strip(), start, end, match.group(0)))
    hits.sort(key=lambda hit: (hit.start, hit.end, hit.name, hit.candidate))
    return tuple(hits)


def generate_slot_candidates(
    text: str, patterns: Iterable[SlotPattern] = SLOT_PATTERNS
) -> tuple[CandidateSpan, ...]:
    """Convert syntax-slot hits into ordinary candidate spans."""

    return tuple(
        CandidateSpan(hit.candidate, hit.start, hit.end, "syntax_slot", (hit.name,))
        for hit in find_slot_hits(text, patterns)
    )


def generate_candidates(
    text: str,
    *,
    min_n: int = 1,
    max_n: int = 5,
    include_runs: bool = True,
    include_slots: bool = True,
) -> tuple[CandidateSpan, ...]:
    """Generate the complete tokenizer-free candidate stream.

    Results are sorted by source span, then source and slot name.  Repeated
    occurrences are retained because scoring needs occurrence counts.
    """

    if not text:
        return ()
    spans = list(generate_char_ngrams(text, min_n, max_n))
    if include_runs:
        spans.extend(generate_latin_kana_digit_runs(text))
    if include_slots:
        spans.extend(generate_slot_candidates(text))
    source_order = {"char_ngram": 0, "latin_kana_digit": 1, "syntax_slot": 2}
    spans.sort(
        key=lambda span: (
            span.start,
            span.end,
            source_order.get(span.source, 99),
            span.slot_names,
            span.text,
        )
    )
    return tuple(spans)


def candidate_strings(text: str, **kwargs: object) -> tuple[str, ...]:
    """Return unique normalized candidate keys in deterministic order."""

    keys = {span.normalized for span in generate_candidates(text, **kwargs) if span.normalized}
    return tuple(sorted(keys, key=lambda value: (len(value), value)))


# Short aliases make the API pleasant in notebooks without hiding the more
# explicit names used by the build pipeline.
char_ngrams = generate_char_ngrams
latin_kana_digit_runs = generate_latin_kana_digit_runs
slot_hits = find_slot_hits


__all__ = [
    "CandidateSpan",
    "SlotHit",
    "SlotPattern",
    "SLOT_PATTERNS",
    "candidate_strings",
    "char_ngrams",
    "find_slot_hits",
    "generate_candidates",
    "generate_char_ngrams",
    "generate_latin_kana_digit_runs",
    "generate_slot_candidates",
    "latin_kana_digit_runs",
    "normalize_for_matching",
    "slot_hits",
]

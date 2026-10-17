"""A pure-Python Aho-Corasick index for entity aliases.

The automaton is intentionally independent of the database layer.  Callers
can pass rows from ``entity_alias`` as mappings, ``AliasRecord`` objects or
simple ``(entity_id, alias_text)`` pairs.  A scan emits every alias match,
selects deterministic non-overlapping spans for erasure, and derives the
unordered entity co-occurrence pairs from the same scan.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence
import unicodedata

from .candidates import normalize_for_matching


@dataclass(frozen=True, slots=True)
class AliasRecord:
    entity_id: str
    alias_text: str
    normalized_alias: str | None = None

    @property
    def alias(self) -> str:
        return self.alias_text


@dataclass(frozen=True, slots=True)
class AliasMatch:
    entity_id: str
    alias: str
    start: int
    end: int
    matched_text: str

    @property
    def span(self) -> tuple[int, int]:
        return self.start, self.end


@dataclass(frozen=True, slots=True)
class EntityCooccurrence:
    entity_id_a: str
    entity_id_b: str

    @property
    def entity_ids(self) -> tuple[str, str]:
        return self.entity_id_a, self.entity_id_b

    @property
    def entities(self) -> tuple[str, str]:
        return self.entity_ids


@dataclass(frozen=True, slots=True)
class ScanResult:
    text: str
    matches: tuple[AliasMatch, ...]
    erased_matches: tuple[AliasMatch, ...]
    erased_text: str
    remaining_text: str
    entity_ids: tuple[str, ...]
    cooccurrence: tuple[EntityCooccurrence, ...]

    @property
    def cooccurrences(self) -> tuple[EntityCooccurrence, ...]:
        return self.cooccurrence

    @property
    def text_without_aliases(self) -> str:
        """Alias for the compact text with known names removed."""

        return self.remaining_text

    @property
    def masked_text(self) -> str:
        """Position-preserving form suitable for a second candidate scan."""

        return self.erased_text


@dataclass(frozen=True, slots=True)
class _Pattern:
    entity_id: str
    alias: str
    normalized: str


@dataclass(slots=True)
class _Node:
    edges: dict[str, int]
    fail: int
    outputs: list[int]


def _coerce_alias(value: object) -> AliasRecord:
    if isinstance(value, AliasRecord):
        return value
    if isinstance(value, Mapping):
        entity_id = value.get("entity_id")
        alias_text = value.get("alias_text", value.get("alias"))
        normalized = value.get("normalized_alias")
        if not isinstance(entity_id, str) or not isinstance(alias_text, str):
            raise TypeError("alias mapping requires string entity_id and alias_text")
        return AliasRecord(entity_id, alias_text, normalized if isinstance(normalized, str) else None)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        entity_id, alias_text = value
        if isinstance(entity_id, str) and isinstance(alias_text, str):
            return AliasRecord(entity_id, alias_text)
    raise TypeError("aliases must be AliasRecord, mapping, or (entity_id, alias_text)")


def _normalise_alias(record: AliasRecord) -> str:
    # Use the explicitly supplied normalized value only after applying the
    # same stable Unicode/case policy as generated values.
    return normalize_for_matching(record.normalized_alias or record.alias_text)


def _normalise_with_offsets(text: str) -> tuple[str, tuple[int, ...]]:
    """Normalize per source code point while retaining source offsets.

    Per-character normalization is sufficient for the aliases in the
    database, preserves useful offsets for compatibility forms such as full
    width Latin, and handles case-fold expansions such as ``ß``.
    """

    pieces: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text):
        piece = unicodedata.normalize("NFKC", char).casefold()
        pieces.append(piece)
        offsets.extend([index] * len(piece))
    return "".join(pieces), tuple(offsets)


class AhoCorasick:
    """Immutable-after-build Aho-Corasick matcher for entity aliases."""

    def __init__(self, aliases: Iterable[object] = ()) -> None:
        records = [_coerce_alias(value) for value in aliases]
        patterns: list[_Pattern] = []
        seen: set[tuple[str, str]] = set()
        for record in records:
            entity_id = record.entity_id.strip()
            alias_text = record.alias_text.strip()
            normalized = _normalise_alias(record)
            if not entity_id or not alias_text or not normalized:
                continue
            key = (entity_id, normalized)
            if key in seen:
                continue
            seen.add(key)
            patterns.append(_Pattern(entity_id, alias_text, normalized))
        patterns.sort(key=lambda pattern: (pattern.normalized, pattern.entity_id, pattern.alias))
        self._patterns = tuple(patterns)
        self._nodes = [_Node({}, 0, [])]
        self._build()

    @classmethod
    def from_aliases(cls, aliases: Iterable[object]) -> AhoCorasick:
        return cls(aliases)

    @property
    def aliases(self) -> tuple[AliasRecord, ...]:
        return tuple(
            AliasRecord(pattern.entity_id, pattern.alias, pattern.normalized)
            for pattern in self._patterns
        )

    def _build(self) -> None:
        for pattern_index, pattern in enumerate(self._patterns):
            state = 0
            for char in pattern.normalized:
                next_state = self._nodes[state].edges.get(char)
                if next_state is None:
                    next_state = len(self._nodes)
                    self._nodes[state].edges[char] = next_state
                    self._nodes.append(_Node({}, 0, []))
                state = next_state
            self._nodes[state].outputs.append(pattern_index)

        queue: deque[int] = deque()
        for child in sorted(self._nodes[0].edges.values()):
            queue.append(child)
            self._nodes[child].fail = 0
        while queue:
            state = queue.popleft()
            for char, child in sorted(self._nodes[state].edges.items()):
                queue.append(child)
                failure = self._nodes[state].fail
                while failure and char not in self._nodes[failure].edges:
                    failure = self._nodes[failure].fail
                self._nodes[child].fail = self._nodes[failure].edges.get(char, 0)
                inherited = self._nodes[self._nodes[child].fail].outputs
                self._nodes[child].outputs.extend(inherited)
                self._nodes[child].outputs.sort()

    def find_matches(self, text: str) -> tuple[AliasMatch, ...]:
        """Return all matches in source-position and alias order."""

        if not text or not self._patterns:
            return ()
        normalized, offsets = _normalise_with_offsets(text)
        if not normalized:
            return ()
        matches: list[AliasMatch] = []
        state = 0
        for normalized_end, char in enumerate(normalized):
            while state and char not in self._nodes[state].edges:
                state = self._nodes[state].fail
            state = self._nodes[state].edges.get(char, 0)
            for pattern_index in self._nodes[state].outputs:
                pattern = self._patterns[pattern_index]
                normalized_start = normalized_end - len(pattern.normalized) + 1
                if normalized_start < 0:
                    continue
                start = offsets[normalized_start]
                end = offsets[normalized_end] + 1
                matches.append(
                    AliasMatch(pattern.entity_id, pattern.alias, start, end, text[start:end])
                )
        matches.sort(key=lambda match: (match.start, match.end, match.entity_id, match.alias))
        return tuple(matches)

    @staticmethod
    def _select_for_erasure(matches: Iterable[AliasMatch]) -> tuple[AliasMatch, ...]:
        """Choose longest-leftmost non-overlapping matches deterministically."""

        selected: list[AliasMatch] = []
        next_free = -1
        ordered = sorted(
            matches,
            key=lambda match: (
                match.start,
                -(match.end - match.start),
                match.entity_id,
                match.alias,
            ),
        )
        for match in ordered:
            if match.start < next_free:
                continue
            selected.append(match)
            next_free = match.end
        selected.sort(key=lambda match: (match.start, match.end, match.entity_id, match.alias))
        return tuple(selected)

    def scan(self, text: str) -> ScanResult:
        """Scan once, erase known aliases, and return entity co-occurrence."""

        if not text:
            return ScanResult("", (), (), "", "", (), ())
        matches = self.find_matches(text)
        entity_ids = tuple(sorted({match.entity_id for match in matches}))
        cooccurrence = tuple(
            EntityCooccurrence(left, right) for left, right in combinations(entity_ids, 2)
        )
        erased_matches = self._select_for_erasure(matches)
        erased = list(text)
        for match in erased_matches:
            for index in range(match.start, match.end):
                erased[index] = " "
        erased_text = "".join(erased)
        remaining_parts: list[str] = []
        cursor = 0
        for match in erased_matches:
            remaining_parts.append(text[cursor : match.start])
            cursor = match.end
        remaining_parts.append(text[cursor:])
        remaining_text = "".join(remaining_parts)
        return ScanResult(
            text,
            matches,
            erased_matches,
            erased_text,
            remaining_text,
            entity_ids,
            cooccurrence,
        )


AhoCorasickAutomaton = AhoCorasick


def compile_entity_aliases(aliases: Iterable[object]) -> AhoCorasick:
    """Compile ``entity_alias``-shaped rows into an AC automaton."""

    return AhoCorasick(aliases)


def scan_entity_aliases(text: str, aliases: Iterable[object]) -> ScanResult:
    """Convenience wrapper for one-off scans."""

    return AhoCorasick(aliases).scan(text)


__all__ = [
    "AhoCorasick",
    "AhoCorasickAutomaton",
    "AliasMatch",
    "AliasRecord",
    "EntityCooccurrence",
    "ScanResult",
    "compile_entity_aliases",
    "scan_entity_aliases",
]

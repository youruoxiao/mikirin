"""Deterministic P1/A4 slang-to-entity linking.

The linker is intentionally a small boundary around evidence that was already
computed by the offline pipeline.  It does not open a database, build an
entity index, embed text, or make network requests.  A vector backend may be
injected, but it is asked for exactly three neighbours and its results are
treated as an external input.

The three buckets are:

``self_reference``
    A candidate concentrated in one subject and used as a replacement for
    that subject's official name.
``cross_work``
    An externally supplied vector-neighbour result, capped at top-3.
``single_char``
    A one-code-point candidate.  It inherits evidence from A/B but has its
    own, deliberately higher thresholds.
"""

from __future__ import annotations

import inspect
import math
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


SELF_REFERENCE = "self_reference"
CROSS_WORK = "cross_work"
SINGLE_CHAR = "single_char"

CONFIRMED = "confirmed"
PROVISIONAL = "provisional"
REJECT = "reject"

_BUCKET_ORDER = {SELF_REFERENCE: 0, CROSS_WORK: 1, SINGLE_CHAR: 2}
_MISSING = object()


def _normalise(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _first(value: Mapping[str, Any], *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if name in value:
            return value[name]
    return default


def _count(value: Any, field: str) -> int:
    if value is None or value is _MISSING:
        return 0
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a non-negative integer") from error
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"{field} must be a non-negative integer")
    return int(number)


def _score(value: Any, field: str = "score") -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return max(0.0, min(1.0, number))


def _as_evidence(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        raw = value.get("evidence", {})
        return dict(raw) if isinstance(raw, Mapping) else {}
    return {}


@dataclass(frozen=True, slots=True)
class AnchorEvidence:
    """Aggregated occurrences of one candidate in one subject.

    ``official_name_count`` is the count of official-name alternatives in the
    same evidence unit (normally documents).  ``None`` means that no official
    name signal was supplied.  A mapping may use the shorter aliases
    ``count``, ``documents`` and ``authors`` when passed to ``link_candidates``.
    """

    subject_entity_id: str
    anchor_count: int
    document_count: int = 0
    author_count: int = 0
    official_name_count: int | None = None
    official_name_document_count: int | None = None
    evidence: Mapping[str, Any] = ()

    def __post_init__(self) -> None:
        if not str(self.subject_entity_id).strip():
            raise ValueError("subject_entity_id must be non-empty")
        if self.anchor_count < 0 or self.document_count < 0 or self.author_count < 0:
            raise ValueError("anchor and breadth counts must be non-negative")
        if self.official_name_count is not None and self.official_name_count < 0:
            raise ValueError("official_name_count must be non-negative")
        if (
            self.official_name_document_count is not None
            and self.official_name_document_count < 0
        ):
            raise ValueError("official_name_document_count must be non-negative")

    @classmethod
    def from_value(cls, subject_entity_id: str, value: Any) -> "AnchorEvidence":
        if isinstance(value, cls):
            if value.subject_entity_id == subject_entity_id:
                return value
            return cls(
                subject_entity_id,
                value.anchor_count,
                value.document_count,
                value.author_count,
                value.official_name_count,
                value.official_name_document_count,
                value.evidence,
            )
        if isinstance(value, Mapping):
            anchor = _first(
                value,
                "anchor_count",
                "occurrence_count",
                "occurrences",
                "mentions",
                "count",
                default=0,
            )
            documents = _first(
                value,
                "document_count",
                "candidate_document_count",
                "documents",
                default=0,
            )
            authors = _first(
                value,
                "author_count",
                "author_breadth",
                "authors",
                default=0,
            )
            official = _first(
                value,
                "official_name_count",
                "official_count",
                "official_name_occurrences",
                default=None,
            )
            official_documents = _first(
                value,
                "official_name_document_count",
                "official_document_count",
                "official_documents",
                default=None,
            )
            return cls(
                subject_entity_id=str(subject_entity_id),
                anchor_count=_count(anchor, "anchor_count"),
                document_count=_count(documents, "document_count"),
                author_count=_count(authors, "author_count"),
                official_name_count=(
                    None if official is None else _count(official, "official_name_count")
                ),
                official_name_document_count=(
                    None
                    if official_documents is None
                    else _count(official_documents, "official_name_document_count")
                ),
                evidence=_as_evidence(value),
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return cls(str(subject_entity_id), _count(value, "anchor_count"))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            values = list(value)
            if not values:
                return cls(str(subject_entity_id), 0)
            return cls(
                str(subject_entity_id),
                _count(values[0], "anchor_count"),
                _count(values[1], "document_count") if len(values) > 1 else 0,
                _count(values[2], "author_count") if len(values) > 2 else 0,
                None,
                None,
            )
        raise ValueError(f"unsupported anchor evidence for {subject_entity_id}")


@dataclass(frozen=True, slots=True)
class VectorNeighbor:
    """One result returned by an external vector search."""

    entity_id: str
    score: float
    evidence: Mapping[str, Any] = ()

    def __post_init__(self) -> None:
        if not str(self.entity_id).strip():
            raise ValueError("entity_id must be non-empty")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")

    @classmethod
    def from_value(cls, value: Any) -> "VectorNeighbor":
        if isinstance(value, cls):
            return cls(str(value.entity_id), _score(value.score), value.evidence)
        if isinstance(value, Mapping):
            entity_id = _first(
                value,
                "entity_id",
                "subject_entity_id",
                "object_id",
                "id",
                default="",
            )
            raw_score = _first(value, "score", "similarity", "cosine", default=0.0)
            if not str(entity_id).strip():
                raise ValueError("vector neighbour entity_id must be non-empty")
            return cls(str(entity_id), _score(raw_score), _as_evidence(value))
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            values = list(value)
            if len(values) == 2:
                entity_id, raw_score = values
                evidence: dict[str, Any] = {}
            elif len(values) >= 3:
                # VectorIndex.search returns (object_kind, object_id, score).
                evidence = {"object_kind": values[0]}
                entity_id, raw_score = values[1], values[2]
            else:
                raise ValueError("vector neighbour tuple must have two or three values")
            if not str(entity_id).strip():
                raise ValueError("vector neighbour entity_id must be non-empty")
            return cls(str(entity_id), _score(raw_score), evidence)
        raise ValueError("unsupported vector neighbour")

    def to_dict(self) -> dict[str, Any]:
        return {"entity_id": self.entity_id, "score": self.score, "evidence": dict(self.evidence)}


class VectorRetriever(Protocol):
    """Minimal optional surface for an injected, already-scoped retriever."""

    def search(self, query: str, *, top_k: int = 3) -> Iterable[Any]: ...


@dataclass(frozen=True, slots=True)
class LinkThresholds:
    """High and medium score cut-offs for each bucket."""

    self_high: float = 0.80
    self_medium: float = 0.60
    cross_high: float = 0.80
    cross_medium: float = 0.65
    single_high: float = 0.95
    single_medium: float = 0.90

    def __post_init__(self) -> None:
        values = (
            self.self_high,
            self.self_medium,
            self.cross_high,
            self.cross_medium,
            self.single_high,
            self.single_medium,
        )
        if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in values):
            raise ValueError("link thresholds must be finite numbers in [0, 1]")
        if self.self_medium > self.self_high or self.cross_medium > self.cross_high:
            raise ValueError("medium threshold cannot exceed high threshold")
        if self.single_medium > self.single_high:
            raise ValueError("single medium threshold cannot exceed single high threshold")

    @classmethod
    def from_value(cls, value: "LinkThresholds | Mapping[str, Any] | None") -> "LinkThresholds":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("thresholds must be LinkThresholds or a mapping")
        aliases = {
            "self_high": ("self_high", "self_reference_high"),
            "self_medium": ("self_medium", "self_reference_medium"),
            "cross_high": ("cross_high", "cross_work_high"),
            "cross_medium": ("cross_medium", "cross_work_medium"),
            "single_high": ("single_high", "single_char_high"),
            "single_medium": ("single_medium", "single_char_medium"),
        }
        values: dict[str, Any] = {}
        for field, names in aliases.items():
            raw = _first(value, *names, default=_MISSING)
            if raw is _MISSING:
                bucket = {
                    "self": SELF_REFERENCE,
                    "cross": CROSS_WORK,
                    "single": SINGLE_CHAR,
                }[field.split("_")[0]]
                nested = value.get(bucket, {})
                if isinstance(nested, Mapping):
                    raw = nested.get(field.split("_")[1], _MISSING)
            if raw is not _MISSING:
                values[field] = _score(raw, field)
        return cls(**values)

    def high(self, bucket: str) -> float:
        return {
            SELF_REFERENCE: self.self_high,
            CROSS_WORK: self.cross_high,
            SINGLE_CHAR: self.single_high,
        }[bucket]

    def medium(self, bucket: str) -> float:
        return {
            SELF_REFERENCE: self.self_medium,
            CROSS_WORK: self.cross_medium,
            SINGLE_CHAR: self.single_medium,
        }[bucket]


@dataclass(frozen=True, slots=True)
class LinkCandidate:
    """A stable, auditable candidate link."""

    candidate_text: str
    entity_id: str
    bucket: str
    rank: int
    score: float
    evidence: Mapping[str, Any]
    candidate_id: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_text.strip() or not self.entity_id.strip():
            raise ValueError("candidate_text and entity_id must be non-empty")
        if self.bucket not in _BUCKET_ORDER:
            raise ValueError(f"unknown link bucket: {self.bucket}")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id or _normalise(self.candidate_text),
            "candidate_text": self.candidate_text,
            "entity_id": self.entity_id,
            "bucket": self.bucket,
            "rank": self.rank,
            "score": self.score,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class IngestionDecision:
    """The write-level decision for one link candidate."""

    decision: str
    effective_weight: float
    candidate_text: str = ""
    entity_id: str | None = None
    bucket: str | None = None
    score: float | None = None
    author_breadth: int = 0
    reason: str = ""

    @property
    def status(self) -> str:
        return self.decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "status": self.status,
            "effective_weight": self.effective_weight,
            "candidate_text": self.candidate_text,
            "entity_id": self.entity_id,
            "bucket": self.bucket,
            "score": self.score,
            "author_breadth": self.author_breadth,
            "reason": self.reason,
        }


def _subject_anchors(
    values: Mapping[str, Any] | Iterable[Any] | None,
) -> tuple[AnchorEvidence, ...]:
    if values is None:
        return ()
    if isinstance(values, AnchorEvidence):
        return (values,)
    if isinstance(values, Mapping):
        result = [
            AnchorEvidence.from_value(str(subject_id), item)
            for subject_id, item in values.items()
        ]
    else:
        result = []
        for item in values:
            if isinstance(item, AnchorEvidence):
                result.append(item)
            elif isinstance(item, Mapping):
                subject_id = _first(
                    item,
                    "subject_entity_id",
                    "subject_id",
                    "entity_id",
                    default="",
                )
                if not str(subject_id).strip():
                    raise ValueError("anchor evidence needs subject_entity_id")
                result.append(AnchorEvidence.from_value(str(subject_id), item))
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                values_list = list(item)
                if len(values_list) < 2:
                    raise ValueError("anchor tuple needs subject_entity_id and count")
                result.append(AnchorEvidence.from_value(str(values_list[0]), values_list[1:]))
            else:
                raise ValueError("unsupported subject anchor evidence")
    # Duplicate subject rows are aggregated, so input order cannot affect output.
    grouped: dict[str, list[AnchorEvidence]] = {}
    for item in result:
        grouped.setdefault(item.subject_entity_id, []).append(item)
    combined: list[AnchorEvidence] = []
    for subject_id in sorted(grouped):
        items = grouped[subject_id]
        explicit_official = any(
            item.official_name_count is not None or item.official_name_document_count is not None
            for item in items
        )
        combined.append(
            AnchorEvidence(
                subject_entity_id=subject_id,
                anchor_count=sum(item.anchor_count for item in items),
                document_count=sum(item.document_count for item in items),
                author_count=sum(item.author_count for item in items),
                official_name_count=(
                    sum(item.official_name_count or 0 for item in items)
                    if explicit_official
                    else None
                ),
                official_name_document_count=(
                    sum(item.official_name_document_count or 0 for item in items)
                    if any(item.official_name_document_count is not None for item in items)
                    else None
                ),
                evidence={
                    key: value
                    for item in items
                    for key, value in dict(item.evidence).items()
                },
            )
        )
    return tuple(combined)


def _official_entry(
    values: Mapping[str, Any] | None, subject_id: str
) -> tuple[bool, int, int, tuple[str, ...]]:
    if not values or subject_id not in values:
        return False, 0, 0, ()
    value = values[subject_id]
    if isinstance(value, Mapping):
        official = _first(
            value,
            "official_name_count",
            "official_count",
            "official_name_occurrences",
            default=None,
        )
        documents = _first(
            value,
            "official_name_document_count",
            "official_document_count",
            "official_documents",
            default=None,
        )
        names = _first(value, "names", "official_names", "canonical_names", default=())
        if isinstance(names, str):
            names = (names,)
        elif not isinstance(names, Sequence):
            names = ()
        return (
            True,
            _count(official, "official_name_count") if official is not None else 0,
            _count(documents, "official_name_document_count") if documents is not None else 0,
            tuple(str(name) for name in names if str(name).strip()),
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True, _count(value, "official_name_count"), 0, ()
    if isinstance(value, str):
        return True, 0, 0, (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return True, 0, 0, tuple(str(name) for name in value if str(name).strip())
    return True, 0, 0, ()


def _call_retriever(retriever: Any, query: str) -> Iterable[Any]:
    method = getattr(retriever, "search", None)
    if method is None:
        method = retriever
    if not callable(method):
        raise ValueError("vector_retriever must be callable or expose search")
    # Prefer the explicit top_k contract.  Signature binding avoids swallowing
    # a TypeError raised inside a backend implementation.
    attempts = (
        ((query,), {"top_k": 3}),
        ((query,), {"limit": 3}),
        ((query, 3), {}),
        ((query,), {}),
    )
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        for args, kwargs in attempts:
            try:
                signature.bind(*args, **kwargs)
            except TypeError:
                continue
            return method(*args, **kwargs)
        raise TypeError("vector_retriever does not accept a query and top-3 limit")
    return method(query, top_k=3)


def _vector_neighbors(
    values: Iterable[Any] | Mapping[str, Any] | None,
) -> tuple[VectorNeighbor, ...]:
    if values is None:
        return ()
    if isinstance(values, Mapping):
        if any(key in values for key in ("entity_id", "subject_entity_id", "object_id", "id")):
            values = (values,)
        else:
            values = values.get("neighbors", values.get("results", ())) or ()
    parsed: dict[str, VectorNeighbor] = {}
    for value in values:
        item = VectorNeighbor.from_value(value)
        old = parsed.get(item.entity_id)
        if old is None or (item.score, item.entity_id) > (old.score, old.entity_id):
            parsed[item.entity_id] = item
    return tuple(
        sorted(parsed.values(), key=lambda item: (-item.score, item.entity_id))[:3]
    )


def _author_value(
    author_breadth: int | Mapping[str, Any] | None,
    entity_id: str,
    evidence: Mapping[str, Any],
) -> int:
    if isinstance(author_breadth, Mapping):
        raw = author_breadth.get(
            entity_id,
            evidence.get("author_breadth", evidence.get("author_count", 0)),
        )
    elif author_breadth is not None:
        raw = author_breadth
    else:
        raw = evidence.get("author_breadth", evidence.get("author_count", 0))
    return _count(raw, "author_breadth")


def _candidate_inputs(
    candidate: str | Mapping[str, Any] | None,
    candidate_text: str | None,
    subject_anchors: Mapping[str, Any] | Iterable[Any] | None,
    official_names: Mapping[str, Any] | None,
    official_name_counts: Mapping[str, Any] | None,
    vector_neighbors: Iterable[Any] | Mapping[str, Any] | None,
) -> tuple[str, str, Any, Mapping[str, Any] | None, Mapping[str, Any] | None, Any]:
    candidate_id = ""
    if isinstance(candidate, Mapping):
        candidate_text = candidate_text or _first(
            candidate, "candidate_text", "text", "candidate", default=""
        )
        candidate_id = str(_first(candidate, "candidate_id", "id", default=""))
        subject_anchors = subject_anchors if subject_anchors is not None else _first(
            candidate, "subject_anchors", "anchors", default=None
        )
        official_names = official_names if official_names is not None else _first(
            candidate, "official_names", default=None
        )
        official_name_counts = official_name_counts if official_name_counts is not None else _first(
            candidate, "official_name_counts", default=None
        )
        vector_neighbors = vector_neighbors if vector_neighbors is not None else _first(
            candidate, "vector_neighbors", "neighbors", default=None
        )
    text = (
        str(candidate_text or candidate or "")
        if not isinstance(candidate, Mapping)
        else str(candidate_text or "")
    )
    return (
        text,
        candidate_id,
        subject_anchors,
        official_names,
        official_name_counts,
        vector_neighbors,
    )


def link_candidates(
    candidate: str | Mapping[str, Any] | None = None,
    *,
    candidate_text: str | None = None,
    subject_anchors: Mapping[str, Any] | Iterable[Any] | None = None,
    anchors: Mapping[str, Any] | Iterable[Any] | None = None,
    official_names: Mapping[str, Any] | None = None,
    official_name_counts: Mapping[str, Any] | None = None,
    vector_neighbors: Iterable[Any] | Mapping[str, Any] | None = None,
    vector_retriever: VectorRetriever | Callable[..., Iterable[Any]] | None = None,
    retriever: VectorRetriever | Callable[..., Iterable[Any]] | None = None,
    author_breadth: int | Mapping[str, Any] | None = None,
    thresholds: LinkThresholds | Mapping[str, Any] | None = None,
    ambiguity_margin: float = 0.10,
) -> tuple[LinkCandidate, ...]:
    """Return deterministic A/B/C link candidates.

    All retrieval evidence is caller-owned.  When ``vector_retriever`` is
    supplied it is called with the candidate text and a top-3 request; its
    result is still capped and sorted locally.  ``vector_neighbors`` takes
    precedence, which makes the function pure for offline builds and tests.
    """

    if anchors is not None:
        if subject_anchors is not None:
            raise ValueError("provide only one of subject_anchors and anchors")
        subject_anchors = anchors
    text, candidate_id, subject_anchors, official_names, official_name_counts, vector_neighbors = (
        _candidate_inputs(
            candidate,
            candidate_text,
            subject_anchors,
            official_names,
            official_name_counts,
            vector_neighbors,
        )
    )
    text = text.strip()
    if not text:
        return ()
    if not 0 <= ambiguity_margin <= 1 or not math.isfinite(float(ambiguity_margin)):
        raise ValueError("ambiguity_margin must be finite and in [0, 1]")
    LinkThresholds.from_value(thresholds)
    anchors_by_subject = _subject_anchors(subject_anchors)
    official_values: dict[str, Any] = {}
    if official_names:
        official_values.update(official_names)
    if official_name_counts:
        for subject_id, value in official_name_counts.items():
            current = official_values.get(subject_id)
            if isinstance(current, Mapping) and isinstance(value, Mapping):
                merged = dict(current)
                merged.update(value)
                official_values[subject_id] = merged
            else:
                official_values[subject_id] = value

    raw: list[tuple[str, str, float, dict[str, Any]]] = []
    total_anchors = sum(item.anchor_count for item in anchors_by_subject)
    if total_anchors:
        ordered = sorted(
            anchors_by_subject,
            key=lambda item: (-item.anchor_count, item.subject_entity_id),
        )
        top_count = ordered[0].anchor_count
        second_count = ordered[1].anchor_count if len(ordered) > 1 else 0
        top_tie = sum(item.anchor_count == top_count for item in ordered) > 1
        close_top = top_count > 0 and (top_count - second_count) / top_count < ambiguity_margin
        for item in ordered:
            if item.anchor_count <= 0 or (
                item.anchor_count != top_count and not close_top
            ):
                continue
            official_signal, official_count, official_documents, names = _official_entry(
                official_values, item.subject_entity_id
            )
            if item.official_name_count is not None:
                official_signal = True
                official_count = item.official_name_count
            if item.official_name_document_count is not None:
                official_signal = True
                official_documents = item.official_name_document_count
            if any(_normalise(name) == _normalise(text) for name in names):
                exclusion = 0.0
                matches_official = True
            else:
                matches_official = False
                candidate_documents = item.document_count or item.anchor_count
                alternative_documents = official_documents or official_count
                exclusion = (
                    candidate_documents / (candidate_documents + alternative_documents)
                    if official_signal and candidate_documents + alternative_documents
                    else 0.0
                )
            concentration = item.anchor_count / total_anchors
            separation = (
                item.anchor_count / (item.anchor_count + second_count)
                if item.anchor_count
                else 0.0
            )
            score = _score(0.55 * concentration + 0.25 * exclusion + 0.20 * separation)
            evidence = {
                "anchor_count": item.anchor_count,
                "document_count": item.document_count,
                "author_breadth": item.author_count,
                "anchor_concentration": concentration,
                "official_name_signal": official_signal,
                "official_name_count": official_count,
                "official_name_document_count": official_documents,
                "official_name_exclusion": exclusion,
                "candidate_matches_official_name": matches_official,
                "ambiguous": bool(top_tie or close_top),
                "top_subject_entity_id": ordered[0].subject_entity_id,
                "top_subject_margin": (top_count - second_count) / top_count,
            }
            evidence.update(dict(item.evidence))
            evidence["author_breadth"] = _author_value(
                author_breadth, item.subject_entity_id, evidence
            )
            raw.append((item.subject_entity_id, SELF_REFERENCE, score, evidence))

    supplied_neighbors = vector_neighbors
    if supplied_neighbors is None and (vector_retriever is not None or retriever is not None):
        supplied_neighbors = _call_retriever(vector_retriever or retriever, text)
    neighbours = _vector_neighbors(supplied_neighbors)
    for index, neighbour in enumerate(neighbours, start=1):
        tied = index == 1 and len(neighbours) > 1 and math.isclose(
            neighbour.score, neighbours[1].score, rel_tol=0.0, abs_tol=1e-12
        )
        evidence = {
            "retrieval": "external_top3",
            "vector_rank": index,
            "vector_score": neighbour.score,
            "ambiguous": tied,
        }
        evidence.update(dict(neighbour.evidence))
        evidence["author_breadth"] = _author_value(
            author_breadth, neighbour.entity_id, evidence
        )
        raw.append((neighbour.entity_id, CROSS_WORK, neighbour.score, evidence))

    if not raw:
        return ()
    is_single_char = len(_normalise(text)) == 1
    if is_single_char:
        raw = [
            (entity_id, SINGLE_CHAR, score, {**evidence, "single_char": True})
            for entity_id, _bucket, score, evidence in raw
        ]

    # A candidate may be produced by both A and B.  The schema has one link row
    # per entity, so retain the strongest evidence and prefer A on exact ties.
    best: dict[str, tuple[str, str, float, dict[str, Any]]] = {}
    for entity_id, bucket, score, evidence in raw:
        current = best.get(entity_id)
        item = (entity_id, bucket, score, evidence)
        if current is None or (score, -_BUCKET_ORDER[bucket], entity_id) > (
            current[2],
            -_BUCKET_ORDER[current[1]],
            entity_id,
        ):
            best[entity_id] = item
        elif current[1] != bucket:
            current[3].setdefault("alternative_buckets", []).append(bucket)
    selected = list(best.values())
    selected.sort(key=lambda item: (-item[2], _BUCKET_ORDER[item[1]], item[0]))
    top_score = selected[0][2]
    top_ids = {
        item[0]
        for item in selected
        if math.isclose(item[2], top_score, rel_tol=0.0, abs_tol=1e-12)
    }
    if len(top_ids) > 1:
        for item in selected:
            if item[0] in top_ids:
                item[3]["ambiguous"] = True
                item[3]["tied_top_entity_ids"] = tuple(sorted(top_ids))
    return tuple(
        LinkCandidate(
            candidate_text=text,
            candidate_id=candidate_id or _normalise(text),
            entity_id=entity_id,
            bucket=bucket,
            rank=rank,
            score=score,
            evidence=evidence,
        )
        for rank, (entity_id, bucket, score, evidence) in enumerate(selected, start=1)
    )


def _as_link_candidate(value: LinkCandidate | Mapping[str, Any]) -> LinkCandidate:
    if isinstance(value, LinkCandidate):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("link candidate must be LinkCandidate or a mapping")
    evidence = value.get("evidence", {})
    if not isinstance(evidence, Mapping):
        evidence = {}
    candidate_id = str(value.get("candidate_id", ""))
    return LinkCandidate(
        candidate_text=str(value.get("candidate_text", value.get("candidate", candidate_id))),
        candidate_id=candidate_id,
        entity_id=str(value.get("entity_id", "")),
        bucket=str(value.get("bucket", value.get("link_bucket", ""))),
        rank=_count(value.get("rank", 1), "rank"),
        score=_score(value.get("score", value.get("link_score", 0.0))),
        evidence=evidence,
    )


def decide_ingestion(
    candidate: LinkCandidate | Mapping[str, Any] | None,
    *,
    author_breadth: int | Mapping[str, Any] | None = None,
    thresholds: LinkThresholds | Mapping[str, Any] | None = None,
) -> IngestionDecision:
    """Apply the A4 write policy to one link candidate.

    High score plus strictly more than 20 authors is ``confirmed``.  A score
    in the medium band is ``provisional`` with effective weight ``0.5``.
    Ambiguous candidates, high scores without enough author breadth, missing A
    official-name evidence, and all other cases are rejected.
    """

    config = LinkThresholds.from_value(thresholds)
    if candidate is None:
        return IngestionDecision(REJECT, 0.0, reason="empty link candidate")
    try:
        item = _as_link_candidate(candidate)
    except (TypeError, ValueError) as error:
        return IngestionDecision(REJECT, 0.0, reason=f"invalid link candidate: {error}")
    breadth = _author_value(author_breadth, item.entity_id, item.evidence)
    if item.evidence.get("ambiguous"):
        reason = "ambiguous link evidence"
    elif item.bucket == SELF_REFERENCE and (
        not item.evidence.get("official_name_signal")
        or item.evidence.get("candidate_matches_official_name")
    ):
        reason = "self-reference lacks mutually exclusive official-name evidence"
    elif item.score >= config.high(item.bucket):
        if breadth > 20:
            return IngestionDecision(
                CONFIRMED,
                1.0,
                item.candidate_text,
                item.entity_id,
                item.bucket,
                item.score,
                breadth,
                "high confidence and author breadth > 20",
            )
        reason = "high confidence but author breadth is not greater than 20"
    elif item.score >= config.medium(item.bucket):
        return IngestionDecision(
            PROVISIONAL,
            0.5,
            item.candidate_text,
            item.entity_id,
            item.bucket,
            item.score,
            breadth,
            "medium confidence",
        )
    else:
        reason = "score below medium threshold"
    return IngestionDecision(
        REJECT,
        0.0,
        item.candidate_text,
        item.entity_id,
        item.bucket,
        item.score,
        breadth,
        reason,
    )


def decide_link_candidates(
    candidates: Iterable[LinkCandidate | Mapping[str, Any]] | None,
    *,
    author_breadth: int | Mapping[str, Any] | None = None,
    thresholds: LinkThresholds | Mapping[str, Any] | None = None,
) -> tuple[IngestionDecision, ...]:
    """Batch form preserving the caller's order; empty input returns ``()``."""

    if candidates is None:
        return ()
    return tuple(
        decide_ingestion(item, author_breadth=author_breadth, thresholds=thresholds)
        for item in candidates
    )


# Explicit aliases keep the API discoverable for callers that use the nouns in
# the schema (``link_score``/``link_bucket``) or the roadmap (``ingest``).
build_link_candidates = link_candidates
ingestion_decision = decide_ingestion
decide_link = decide_ingestion


__all__ = [
    "AnchorEvidence",
    "CROSS_WORK",
    "CONFIRMED",
    "IngestionDecision",
    "LinkCandidate",
    "LinkThresholds",
    "PROVISIONAL",
    "REJECT",
    "SELF_REFERENCE",
    "SINGLE_CHAR",
    "VectorNeighbor",
    "VectorRetriever",
    "build_link_candidates",
    "decide_ingestion",
    "decide_link",
    "decide_link_candidates",
    "ingestion_decision",
    "link_candidates",
]

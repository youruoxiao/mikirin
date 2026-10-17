"""Pure-statistics features, scoring and deterministic link buckets for A4."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping

from .candidates import CandidateSpan, generate_candidates, normalize_for_matching


BOS = "<BOS>"
EOS = "<EOS>"
NONE_CONTEXT = "<NONE>"


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    document_id: str
    text: str
    author_id: str | None = None
    context: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    """One candidate occurrence with the evidence needed by the scorer."""

    candidate: str
    document_id: str
    author_id: str | None = None
    left: str = ""
    right: str = ""
    context: str | None = None
    slot_names: tuple[str, ...] = ()
    title_bracketed: bool = False
    source_text: str = ""

    @property
    def normalized_candidate(self) -> str:
        return normalize_for_matching(self.candidate)


def _is_anonymous(author_id: str | None) -> bool:
    if not author_id or not author_id.strip():
        return True
    return author_id.casefold().strip() in {"anonymous", "anon", "匿名"}


_TITLE_BRACKETS: tuple[tuple[str, str], ...] = (
    ("《", "》"),
    ("〈", "〉"),
    ("「", "」"),
    ("『", "』"),
    ("【", "】"),
    ("（", "）"),
    ("(", ")"),
    ("[", "]"),
    ("“", "”"),
    ('"', '"'),
    ("'", "'"),
)


def is_title_bracketed(text: str, start: int, end: int) -> bool:
    """Return whether a span is directly enclosed by a title-like pair."""

    if start < 1 or end >= len(text):
        return False
    left, right = text[start - 1], text[end]
    return (left, right) in _TITLE_BRACKETS


def observations_from_documents(
    documents: Iterable[DocumentRecord],
    *,
    min_n: int = 1,
    max_n: int = 5,
    include_runs: bool = True,
    include_slots: bool = True,
) -> tuple[CandidateObservation, ...]:
    """Generate scored occurrences from documents using the A4 candidate API."""

    observations: list[CandidateObservation] = []
    for document in documents:
        if not document.text:
            continue
        spans = generate_candidates(
            document.text,
            min_n=min_n,
            max_n=max_n,
            include_runs=include_runs,
            include_slots=include_slots,
        )
        for span in spans:
            left = document.text[span.start - 1] if span.start else BOS
            right = document.text[span.end] if span.end < len(document.text) else EOS
            observations.append(
                CandidateObservation(
                    candidate=span.text,
                    document_id=document.document_id,
                    author_id=document.author_id,
                    left=left,
                    right=right,
                    context=document.context,
                    slot_names=span.slot_names,
                    title_bracketed=is_title_bracketed(document.text, span.start, span.end),
                    source_text=document.text,
                )
            )
    return tuple(observations)


def _sorted_counter(counter: Mapping[str, int]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _sorted_nested_counter(
    counter: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    return {key: _sorted_counter(counter[key]) for key in sorted(counter)}


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Deterministic sufficient statistics for candidate feature extraction."""

    total_documents: int
    candidate_occurrences: Mapping[str, int]
    candidate_documents: Mapping[str, int]
    left_neighbors: Mapping[str, Mapping[str, int]]
    right_neighbors: Mapping[str, Mapping[str, int]]
    authors: Mapping[str, tuple[str, ...]]
    contexts: Mapping[str, Mapping[str, int]]
    slot_hits: Mapping[str, int]
    bracketed_occurrences: Mapping[str, int]
    unigram_counts: Mapping[str, int]
    bigram_counts: Mapping[str, int]
    total_characters: int
    total_authors: int
    ngram_counts: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_observations(cls, observations: Iterable[CandidateObservation]) -> CorpusStats:
        return build_corpus_stats(observations)


def _count_text_ngrams(
    text: str, unigrams: dict[str, int], bigrams: dict[str, int],
    ngrams: dict[str, int] | None = None,
) -> int:
    normalized = normalize_for_matching(text)
    total = 0
    for run_start, run_end in _non_space_runs(normalized):
        chars = normalized[run_start:run_end]
        total += len(chars)
        for char in chars:
            unigrams[char] = unigrams.get(char, 0) + 1
        for index in range(len(chars) - 1):
            pair = chars[index : index + 2]
            bigrams[pair] = bigrams.get(pair, 0) + 1
        if ngrams is not None:
            for width in range(1, 6):
                for index in range(len(chars) - width + 1):
                    gram = chars[index:index + width]
                    ngrams[gram] = ngrams.get(gram, 0) + 1
    return total


def _non_space_runs(text: str) -> Iterable[tuple[int, int]]:
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


def build_corpus_stats(observations: Iterable[CandidateObservation]) -> CorpusStats:
    """Aggregate observations with no dependence on input iteration order."""

    rows = [row for row in observations if row.normalized_candidate]
    rows.sort(
        key=lambda row: (
            row.document_id,
            row.normalized_candidate,
            row.left,
            row.right,
            row.context or NONE_CONTEXT,
            row.author_id or "",
            row.title_bracketed,
            row.slot_names,
        )
    )

    occurrences: dict[str, int] = {}
    candidate_docs: dict[str, set[str]] = {}
    left_neighbors: dict[str, dict[str, int]] = {}
    right_neighbors: dict[str, dict[str, int]] = {}
    authors: dict[str, set[str]] = {}
    contexts_seen: set[tuple[str, str, str]] = set()
    slot_hits: dict[str, int] = {}
    bracketed: dict[str, int] = {}
    unigrams: dict[str, int] = {}
    bigrams: dict[str, int] = {}
    ngrams: dict[str, int] = {}

    # A document's full text is counted once even though it produces many
    # candidate rows.  Sorting makes the fallback for duplicate IDs stable.
    source_by_document: dict[str, str] = {}
    for row in rows:
        source = row.source_text or row.candidate
        previous = source_by_document.get(row.document_id)
        if previous is None or source < previous:
            source_by_document[row.document_id] = source
    total_characters = 0
    for document_id in sorted(source_by_document):
        total_characters += _count_text_ngrams(source_by_document[document_id], unigrams, bigrams, ngrams)

    # Long run/slot candidates also need their full word and every prefix/suffix,
    # not all O(n^2) substrings of the corpus.
    needed = {part for row in rows if len(row.normalized_candidate) > 5
              for split in range(1, len(row.normalized_candidate))
              for part in (row.normalized_candidate, row.normalized_candidate[:split],
                           row.normalized_candidate[split:]) if len(part) > 5}
    for part in sorted(needed):
        ngrams[part] = sum(sum(text.startswith(part, index) for index in range(len(text)))
                           for text in map(normalize_for_matching, source_by_document.values()))

    for row in rows:
        key = row.normalized_candidate
        occurrences[key] = occurrences.get(key, 0) + 1
        candidate_docs.setdefault(key, set()).add(row.document_id)
        left = row.left or BOS
        right = row.right or EOS
        left_counts = left_neighbors.setdefault(key, {})
        right_counts = right_neighbors.setdefault(key, {})
        left_counts[left] = left_counts.get(left, 0) + 1
        right_counts[right] = right_counts.get(right, 0) + 1
        if not _is_anonymous(row.author_id):
            authors.setdefault(key, set()).add(row.author_id.strip())
        contexts_seen.add((key, row.document_id, row.context or NONE_CONTEXT))
        if row.slot_names:
            slot_hits[key] = slot_hits.get(key, 0) + len(row.slot_names)
        if row.title_bracketed:
            bracketed[key] = bracketed.get(key, 0) + 1

    contexts: dict[str, dict[str, int]] = {}
    for key, document_id, context in sorted(contexts_seen):
        context_counts = contexts.setdefault(key, {})
        context_counts[context] = context_counts.get(context, 0) + 1
    author_values = {key: tuple(sorted(values)) for key, values in authors.items()}
    total_authors = len({author for values in author_values.values() for author in values})
    return CorpusStats(
        total_documents=len(source_by_document),
        candidate_occurrences=_sorted_counter(occurrences),
        candidate_documents={key: len(candidate_docs[key]) for key in sorted(candidate_docs)},
        left_neighbors=_sorted_nested_counter(left_neighbors),
        right_neighbors=_sorted_nested_counter(right_neighbors),
        authors={key: author_values[key] for key in sorted(author_values)},
        contexts=_sorted_nested_counter(contexts),
        slot_hits=_sorted_counter(slot_hits),
        bracketed_occurrences=_sorted_counter(bracketed),
        unigram_counts=_sorted_counter(unigrams),
        bigram_counts=_sorted_counter(bigrams),
        total_characters=total_characters,
        total_authors=total_authors,
        ngram_counts=_sorted_counter(ngrams),
    )


def _entropy(counts: Mapping[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values() if count)


def _min_pmi(candidate: str, stats: CorpusStats) -> float:
    if len(candidate) < 2 or stats.total_characters <= 0:
        return 0.0
    values: list[float] = []
    for index in range(1, len(candidate)):
        left, right = candidate[:index], candidate[index:]
        counts = stats.ngram_counts or {**stats.unigram_counts, **stats.bigram_counts,
                                       **stats.candidate_occurrences}
        left_count = counts.get(left, 0)
        right_count = counts.get(right, 0)
        pair_count = counts.get(candidate, 0)
        if not left_count or not right_count or not pair_count:
            # Keep the feature JSON-friendly for sparse or externally supplied
            # observations while retaining a clearly poor PMI signal.
            return -20.0
        values.append(
            math.log2(
                (pair_count * stats.total_characters)
                / (left_count * right_count)
            )
        )
    return min(values, default=0.0)


@dataclass(frozen=True, slots=True)
class FeatureVector:
    candidate: str
    occurrence_count: int
    document_frequency: int
    min_pmi: float
    left_adjacency_entropy: float
    right_adjacency_entropy: float
    author_breadth: int
    missing_title_bracket_rate: float
    context_distribution_entropy: float
    slot_hits: int
    slot_hit_rate: float
    left_neighbor_cardinality: int = 0
    right_neighbor_cardinality: int = 0
    context_cardinality: int = 0

    @property
    def adjacency_entropy(self) -> float:
        return (self.left_adjacency_entropy + self.right_adjacency_entropy) / 2.0

    @property
    def pmi(self) -> float:
        return self.min_pmi

    @property
    def left_entropy(self) -> float:
        return self.left_adjacency_entropy

    @property
    def right_entropy(self) -> float:
        return self.right_adjacency_entropy

    @property
    def context_entropy(self) -> float:
        return self.context_distribution_entropy

    @property
    def bracket_missing_rate(self) -> float:
        return self.missing_title_bracket_rate

    @property
    def slot_hit_count(self) -> int:
        return self.slot_hits

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "candidate": self.candidate,
            "occurrence_count": self.occurrence_count,
            "document_frequency": self.document_frequency,
            "min_pmi": self.min_pmi,
            "left_adjacency_entropy": self.left_adjacency_entropy,
            "right_adjacency_entropy": self.right_adjacency_entropy,
            "adjacency_entropy": self.adjacency_entropy,
            "author_breadth": self.author_breadth,
            "missing_title_bracket_rate": self.missing_title_bracket_rate,
            "context_distribution_entropy": self.context_distribution_entropy,
            "slot_hits": self.slot_hits,
            "slot_hit_rate": self.slot_hit_rate,
        }


def compute_features(candidate: str, stats: CorpusStats) -> FeatureVector:
    """Compute all A4 deterministic features for one candidate key."""

    key = normalize_for_matching(candidate)
    occurrence_count = stats.candidate_occurrences.get(key, 0)
    left = stats.left_neighbors.get(key, {})
    right = stats.right_neighbors.get(key, {})
    contexts = stats.contexts.get(key, {})
    slot_count = stats.slot_hits.get(key, 0)
    bracketed_count = stats.bracketed_occurrences.get(key, 0)
    missing_rate = 0.0
    if occurrence_count:
        missing_rate = max(0.0, min(1.0, 1.0 - bracketed_count / occurrence_count))
    slot_rate = min(1.0, slot_count / occurrence_count) if occurrence_count else 0.0
    return FeatureVector(
        candidate=key,
        occurrence_count=occurrence_count,
        document_frequency=stats.candidate_documents.get(key, 0),
        min_pmi=_min_pmi(key, stats),
        left_adjacency_entropy=_entropy(left),
        right_adjacency_entropy=_entropy(right),
        author_breadth=len(stats.authors.get(key, ())),
        missing_title_bracket_rate=missing_rate,
        context_distribution_entropy=_entropy(contexts),
        slot_hits=slot_count,
        slot_hit_rate=slot_rate,
        left_neighbor_cardinality=len(left),
        right_neighbor_cardinality=len(right),
        context_cardinality=len(contexts),
    )


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    min_pmi: float = 0.20
    left_adjacency_entropy: float = 0.10
    right_adjacency_entropy: float = 0.10
    author_breadth: float = 0.10
    missing_title_bracket_rate: float = 0.10
    context_distribution_entropy: float = 0.15
    slot_hit_rate: float = 0.25

    def as_dict(self) -> dict[str, float]:
        return {
            "min_pmi": self.min_pmi,
            "left_adjacency_entropy": self.left_adjacency_entropy,
            "right_adjacency_entropy": self.right_adjacency_entropy,
            "author_breadth": self.author_breadth,
            "missing_title_bracket_rate": self.missing_title_bracket_rate,
            "context_distribution_entropy": self.context_distribution_entropy,
            "slot_hit_rate": self.slot_hit_rate,
        }


@dataclass(frozen=True, slots=True)
class ScoreBuckets:
    low: float = 0.35
    medium: float = 0.60
    high: float = 0.80

    def __post_init__(self) -> None:
        if not 0.0 <= self.low <= self.medium <= self.high <= 1.0:
            raise ValueError("bucket thresholds must satisfy 0 <= low <= medium <= high <= 1")


class ScoreBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


def bucket_score(score: float, buckets: ScoreBuckets = ScoreBuckets()) -> ScoreBand:
    """Put a [0, 1] score into a stable four-band bucket."""

    value = max(0.0, min(1.0, float(score)))
    if value < buckets.low:
        return ScoreBand.LOW
    if value < buckets.medium:
        return ScoreBand.MEDIUM
    if value < buckets.high:
        return ScoreBand.HIGH
    return ScoreBand.VERY_HIGH


def _bounded_entropy(value: float, cardinality: int) -> float:
    if cardinality <= 1:
        return 0.0
    return max(0.0, min(1.0, value / math.log2(cardinality)))


def _bounded_pmi(value: float) -> float:
    if math.isnan(value):
        return 0.0
    if value == float("inf"):
        return 1.0
    if value == float("-inf"):
        return 0.0
    return max(0.0, min(1.0, 0.5 + 0.5 * math.tanh(value / 4.0)))


def _bounded_author_breadth(value: int, total_authors: int) -> float:
    if value <= 0:
        return 0.0
    denominator = math.log1p(max(20, total_authors))
    return min(1.0, math.log1p(value) / denominator)


@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate: str
    features: FeatureVector
    normalized_features: Mapping[str, float]
    weighted_components: Mapping[str, float]
    score: float
    bucket: ScoreBand

    @property
    def value(self) -> float:
        return self.score


def score_candidate(
    candidate: str,
    stats: CorpusStats,
    *,
    weights: ScoreWeights = ScoreWeights(),
    buckets: ScoreBuckets = ScoreBuckets(),
) -> CandidateScore:
    """Compute a reproducible weighted [0, 1] score and its bucket."""

    features = compute_features(candidate, stats)
    normalized = {
        "min_pmi": _bounded_pmi(features.min_pmi),
        "left_adjacency_entropy": _bounded_entropy(
            features.left_adjacency_entropy, features.left_neighbor_cardinality
        ),
        "right_adjacency_entropy": _bounded_entropy(
            features.right_adjacency_entropy, features.right_neighbor_cardinality
        ),
        "author_breadth": _bounded_author_breadth(features.author_breadth, stats.total_authors),
        "missing_title_bracket_rate": features.missing_title_bracket_rate,
        "context_distribution_entropy": _bounded_entropy(
            features.context_distribution_entropy, features.context_cardinality
        ),
        "slot_hit_rate": features.slot_hit_rate,
    }
    raw_weights = weights.as_dict()
    positive_weights = {key: max(0.0, value) for key, value in raw_weights.items()}
    total_weight = sum(positive_weights.values())
    weighted = {
        key: normalized[key] * positive_weights[key] for key in sorted(normalized)
    }
    score = sum(weighted.values()) / total_weight if total_weight else 0.0
    if not features.occurrence_count:
        score = 0.0
    score = max(0.0, min(1.0, score))
    return CandidateScore(
        candidate=features.candidate,
        features=features,
        normalized_features={key: normalized[key] for key in sorted(normalized)},
        weighted_components=weighted,
        score=score,
        bucket=bucket_score(score, buckets),
    )


def rank_candidates(
    candidates: Iterable[str],
    stats: CorpusStats,
    *,
    weights: ScoreWeights = ScoreWeights(),
    buckets: ScoreBuckets = ScoreBuckets(),
) -> tuple[CandidateScore, ...]:
    """Score candidates and sort by score descending then key ascending."""

    unique = {normalize_for_matching(candidate) for candidate in candidates if normalize_for_matching(candidate)}
    scores = [score_candidate(candidate, stats, weights=weights, buckets=buckets) for candidate in unique]
    scores.sort(key=lambda item: (-item.score, item.candidate))
    return tuple(scores)


class LinkBucket(str, Enum):
    A_SELF = "A_SELF"
    B_CROSS_WORK = "B_CROSS_WORK"
    C_SINGLE_CHAR = "C_SINGLE_CHAR"
    REVIEW = "REVIEW"


@dataclass(frozen=True, slots=True)
class BucketEvidence:
    """Evidence supplied by later linking stages, kept separate from score."""

    self_anchor_concentration: float = 0.0
    official_alias_exclusive: bool = False
    cross_work_hits: int = 0


def link_bucket(
    candidate: str,
    score: CandidateScore,
    evidence: BucketEvidence = BucketEvidence(),
    *,
    self_concentration_threshold: float = 0.50,
    single_char_score_threshold: float = 0.85,
) -> LinkBucket:
    """Apply conservative A/B/C linking rules after statistical scoring."""

    key = normalize_for_matching(candidate)
    if len(key) == 1:
        return (
            LinkBucket.C_SINGLE_CHAR
            if score.score >= single_char_score_threshold
            else LinkBucket.REVIEW
        )
    if (
        evidence.official_alias_exclusive
        and evidence.self_anchor_concentration >= self_concentration_threshold
    ):
        return LinkBucket.A_SELF
    if evidence.cross_work_hits > 0:
        return LinkBucket.B_CROSS_WORK
    return LinkBucket.REVIEW


# Common descriptive aliases for callers that prefer function-style names.
build_statistics = build_corpus_stats
extract_features = compute_features
score = score_candidate
assign_link_bucket = link_bucket


__all__ = [
    "BOS",
    "EOS",
    "BucketEvidence",
    "CandidateObservation",
    "CandidateScore",
    "CorpusStats",
    "DocumentRecord",
    "FeatureVector",
    "LinkBucket",
    "NONE_CONTEXT",
    "ScoreBand",
    "ScoreBuckets",
    "ScoreWeights",
    "assign_link_bucket",
    "build_corpus_stats",
    "build_statistics",
    "bucket_score",
    "compute_features",
    "extract_features",
    "is_title_bracketed",
    "link_bucket",
    "observations_from_documents",
    "rank_candidates",
    "score",
    "score_candidate",
]

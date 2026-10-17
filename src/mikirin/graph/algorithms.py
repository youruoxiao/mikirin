"""Pure, deterministic graph helpers for A5.

The module deliberately contains no database or model integration.  Callers can
build graph rows offline and materialize the returned :class:`GraphEdge`
objects as ``derived_edge``/``implicit_edge`` rows later.

All collection-returning functions use sorted keys and never depend on the
iteration order of their inputs.  Entity and tag identifiers are treated as
strings; blank identifiers are ignored at the input boundary.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import fsum, isfinite, log, sqrt
from typing import Iterable, Mapping, TypeAlias


NodeId: TypeAlias = str
Score: TypeAlias = tuple[NodeId, float]


class PageRankConvergenceError(RuntimeError):
    """Raised when PageRank exhausts its iteration budget without converging."""

    def __init__(
        self,
        *,
        iterations: int,
        max_iterations: int,
        tolerance: float,
        last_delta: float,
    ) -> None:
        self.iterations = iterations
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.last_delta = last_delta
        super().__init__(
            "PageRank did not converge after "
            f"{iterations} iterations (max_iterations={max_iterations}, "
            f"last_delta={last_delta:.17g}, tolerance={tolerance:.17g})"
        )


@dataclass(frozen=True, slots=True)
class GraphNode:
    """A graph node identified by a stable entity id."""

    node_id: NodeId
    kind: str = "subject"

    @property
    def entity_id(self) -> NodeId:
        return self.node_id


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """A weighted graph edge.

    Derived overlap and tag edges are undirected by default.  ``directed`` is
    available for implicit relations whose source/target direction matters.
    """

    source: NodeId
    target: NodeId
    weight: float
    kind: str = ""
    directed: bool = False

    @property
    def src_entity_id(self) -> NodeId:
        return self.source

    @property
    def dst_entity_id(self) -> NodeId:
        return self.target


# Short names are useful to offline callers, while the explicit names make the
# public shape of graph rows easy to discover.
Node = GraphNode
Edge = GraphEdge


def _text(value: object) -> str:
    """Return a stable, trimmed string representation for an input id."""

    if value is None:
        return ""
    return str(value).strip()


def _unique_members(values: object) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, (str, bytes)):
        values = (values,)
    try:
        return {_text(value) for value in values if _text(value)}  # type: ignore[union-attr]
    except TypeError:
        value = _text(values)
        return {value} if value else set()


def _subject_memberships(
    subject_members: Mapping[object, object] | Iterable[tuple[object, object]],
) -> dict[NodeId, set[str]]:
    """Normalize subject -> member collections, merging repeated rows."""

    normalized: dict[NodeId, set[str]] = defaultdict(set)
    if isinstance(subject_members, Mapping):
        rows = subject_members.items()
    else:
        rows = subject_members
    for subject, members in rows:
        subject_id = _text(subject)
        if not subject_id:
            continue
        normalized[subject_id].update(_unique_members(members))
    return dict(normalized)


def _canonical_pair(source: object, target: object) -> tuple[NodeId, NodeId]:
    source_id, target_id = _text(source), _text(target)
    return (source_id, target_id) if source_id <= target_id else (target_id, source_id)


def tag_idf_weights(subject_tags: Mapping[object, object]) -> dict[str, float]:
    """Return corpus IDF weights for tags.

    Document frequency counts a tag at most once per subject.  The formula is
    ``log(number_of_subjects / document_frequency)``; therefore a tag present
    on every subject receives zero weight, preventing generic tags from
    becoming high-degree similarity hubs.
    """

    documents = _subject_memberships(subject_tags)  # type: ignore[arg-type]
    subject_count = len(documents)
    if subject_count == 0:
        return {}

    document_frequency: dict[str, int] = defaultdict(int)
    for tags in documents.values():
        for tag in sorted(tags):
            document_frequency[tag] += 1
    return {
        tag: log(subject_count / frequency)
        for tag, frequency in sorted(document_frequency.items())
    }


compute_tag_idf = tag_idf_weights


def tag_similarity(
    tags_a: Iterable[object],
    tags_b: Iterable[object],
    idf_weights: Mapping[object, float] | None = None,
) -> float:
    """Calculate cosine similarity of two IDF-weighted tag vectors.

    When no corpus weights are supplied, this is ordinary binary cosine
    similarity.  A supplied IDF mapping is used as the vector component;
    unknown tags have zero weight, which keeps the corpus contract explicit.
    """

    left = _unique_members(tags_a)
    right = _unique_members(tags_b)
    if not left or not right:
        return 0.0

    if idf_weights is None:
        weights = {tag: 1.0 for tag in left | right}
    else:
        weights = {}
        for tag in left | right:
            weight = float(idf_weights.get(tag, 0.0))
            if not isfinite(weight) or weight < 0:
                raise ValueError("IDF weights must be finite and non-negative")
            weights[tag] = weight

    left_norm = sqrt(sum(weights[tag] * weights[tag] for tag in left))
    right_norm = sqrt(sum(weights[tag] * weights[tag] for tag in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    common = sorted(left & right)
    numerator = sum(weights[tag] * weights[tag] for tag in common)
    return numerator / (left_norm * right_norm)


weighted_tag_similarity = tag_similarity


def subject_tag_similarities(
    subject_tags: Mapping[object, object],
    idf_weights: Mapping[object, float] | None = None,
) -> dict[tuple[NodeId, NodeId], float]:
    """Return IDF-weighted similarities for every unordered subject pair."""

    documents = _subject_memberships(subject_tags)  # type: ignore[arg-type]
    weights = tag_idf_weights(documents) if idf_weights is None else dict(idf_weights)
    subjects = sorted(documents)
    similarities: dict[tuple[NodeId, NodeId], float] = {}
    for index, source in enumerate(subjects):
        for target in subjects[index + 1 :]:
            similarities[(source, target)] = tag_similarity(
                documents[source], documents[target], weights
            )
    return similarities


pairwise_tag_similarity = subject_tag_similarities
compute_tag_similarities = subject_tag_similarities


def tag_similarity_edges(
    subject_tags: Mapping[object, object],
    idf_weights: Mapping[object, float] | None = None,
    *,
    min_weight: float = 0.0,
) -> tuple[GraphEdge, ...]:
    """Convert pairwise tag similarities into deterministic derived edges."""

    if not isfinite(float(min_weight)) or min_weight < 0:
        raise ValueError("min_weight must be finite and non-negative")
    similarities = subject_tag_similarities(subject_tags, idf_weights)
    return tuple(
        GraphEdge(source, target, weight, "tag_similarity")
        for (source, target), weight in similarities.items()
        if weight >= min_weight and weight > 0.0
    )


def aggregate_overlap_weights(
    subject_members: Mapping[object, object] | Iterable[tuple[object, object]],
) -> dict[tuple[NodeId, NodeId], int]:
    """Count shared unique members for every unordered subject pair."""

    memberships = _subject_memberships(subject_members)
    subjects_by_member: dict[str, list[NodeId]] = defaultdict(list)
    for subject in sorted(memberships):
        for member in sorted(memberships[subject]):
            subjects_by_member[member].append(subject)

    weights: dict[tuple[NodeId, NodeId], int] = defaultdict(int)
    for subjects in subjects_by_member.values():
        for index, source in enumerate(subjects):
            for target in subjects[index + 1 :]:
                weights[(source, target)] += 1
    return dict(sorted(weights.items()))


def aggregate_overlap_edges(
    subject_members: Mapping[object, object] | Iterable[tuple[object, object]],
    *,
    kind: str,
) -> tuple[GraphEdge, ...]:
    """Build weighted overlap edges with a caller-supplied derived-edge kind."""

    edge_kind = _text(kind)
    if not edge_kind:
        raise ValueError("edge kind must not be blank")
    return tuple(
        GraphEdge(source, target, float(weight), edge_kind)
        for (source, target), weight in aggregate_overlap_weights(subject_members).items()
    )


def aggregate_staff_overlap(
    subject_staff: Mapping[object, object] | Iterable[tuple[object, object]],
) -> tuple[GraphEdge, ...]:
    """Aggregate shared staff ids into ``staff_overlap`` edges."""

    return aggregate_overlap_edges(subject_staff, kind="staff_overlap")


def aggregate_cv_overlap(
    subject_cv: Mapping[object, object] | Iterable[tuple[object, object]],
) -> tuple[GraphEdge, ...]:
    """Aggregate shared voice-actor ids into ``cv_overlap`` edges."""

    return aggregate_overlap_edges(subject_cv, kind="cv_overlap")


staff_overlap_edges = aggregate_staff_overlap
cv_overlap_edges = aggregate_cv_overlap


_ANONYMOUS_AUTHORS = frozenset(
    {"anonymous", "anonymous-user", "anonymous_user", "anon", "匿名"}
)


def _author_hash(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key in ("author_hash", "authorHash", "author"):
            if key in value:
                value = value[key]
                break
    if value is None:
        return None
    author = _text(value)
    if not author or author.casefold() in _ANONYMOUS_AUTHORS:
        return None
    return author


def count_distinct_author_hashes(author_hashes: Iterable[object]) -> int:
    """Count unique non-empty, non-anonymous author hashes."""

    return len(
        {
            author
            for value in author_hashes
            if (author := _author_hash(value)) is not None
        }
    )


count_distinct_authors = count_distinct_author_hashes
implicit_edge_weight = count_distinct_author_hashes


def _record_value(record: Mapping[object, object], *names: str) -> object | None:
    for name in names:
        if name in record:
            return record[name]
    return None


def aggregate_implicit_edge_weights(
    observations: Mapping[object, object] | Iterable[object],
) -> dict[tuple[NodeId, NodeId], int]:
    """Aggregate implicit edge weights by canonical source/target pair.

    Accepted inputs are either ``{(src, dst): authors}`` or records containing
    ``src``/``dst`` (also ``source``/``target`` and ``*_entity_id``) plus an
    ``author_hash`` field.  This helper is optional for Pass B, but keeps its
    ``COUNT(DISTINCT author_hash)`` rule in one pure function.
    """

    authors_by_edge: dict[tuple[NodeId, NodeId], set[str]] = defaultdict(set)
    if isinstance(observations, Mapping):
        for pair, authors in observations.items():
            try:
                source, target = pair  # type: ignore[misc]
            except (TypeError, ValueError):
                raise ValueError("implicit edge mapping keys must be (source, target)") from None
            edge = _canonical_pair(source, target)
            if not all(edge):
                continue
            values = authors if not isinstance(authors, (str, bytes)) else (authors,)
            for value in values:  # type: ignore[union-attr]
                if (author := _author_hash(value)) is not None:
                    authors_by_edge[edge].add(author)
    else:
        for observation in observations:
            if isinstance(observation, Mapping):
                source = _record_value(
                    observation, "src", "source", "src_entity_id", "source_entity_id"
                )
                target = _record_value(
                    observation, "dst", "target", "dst_entity_id", "target_entity_id"
                )
                author = _record_value(observation, "author_hash", "authorHash", "author")
            else:
                try:
                    source, target, author = observation  # type: ignore[misc]
                except (TypeError, ValueError):
                    raise ValueError(
                        "implicit observations must contain source, target, author"
                    ) from None
            edge = _canonical_pair(source, target)
            if all(edge) and (author_value := _author_hash(author)) is not None:
                authors_by_edge[edge].add(author_value)
    return {
        edge: len(authors)
        for edge, authors in sorted(authors_by_edge.items())
        if authors
    }


def _node_value(value: object) -> NodeId:
    if isinstance(value, GraphNode):
        return _text(value.node_id)
    return _text(value)


def _edge_values(edge: object) -> tuple[NodeId, NodeId, float, bool]:
    if isinstance(edge, GraphEdge):
        source, target, weight, directed = edge.source, edge.target, edge.weight, edge.directed
    elif isinstance(edge, Mapping):
        source = edge.get("source", edge.get("src", edge.get("src_entity_id")))
        target = edge.get("target", edge.get("dst", edge.get("dst_entity_id")))
        weight = edge.get("weight", 0.0)
        directed = bool(edge.get("directed", False))
    else:
        try:
            values = tuple(edge)  # type: ignore[arg-type]
        except TypeError:
            raise TypeError("edges must be GraphEdge, mapping, or a tuple") from None
        if len(values) not in {3, 4, 5}:
            raise ValueError("edge tuples must have 3 to 5 values")
        source, target, weight = values[:3]
        if len(values) == 4:
            directed = values[3] if isinstance(values[3], bool) else False
        else:
            directed = bool(values[4] if len(values) == 5 else False)
    source_id, target_id = _node_value(source), _node_value(target)
    if not source_id or not target_id:
        raise ValueError("edge endpoints must not be blank")
    numeric_weight = float(weight)
    if not isfinite(numeric_weight) or numeric_weight < 0.0:
        raise ValueError("edge weights must be finite and non-negative")
    return source_id, target_id, numeric_weight, directed


def _normalize_personalization(
    personalization: Mapping[object, float] | Iterable[object] | object,
    nodes: tuple[NodeId, ...],
) -> dict[NodeId, float]:
    raw: dict[NodeId, float] = defaultdict(float)
    if isinstance(personalization, Mapping):
        values = personalization.items()
    elif isinstance(personalization, (str, GraphNode)):
        values = ((_node_value(personalization), 1.0),)
    else:
        try:
            values = (
                (_node_value(value), 1.0)
                for value in personalization  # type: ignore[arg-type]
            )
        except TypeError:
            values = ((_node_value(personalization), 1.0),)
    node_set = set(nodes)
    for node, value in values:
        node_id = _node_value(node)
        numeric_value = float(value)
        if not isfinite(numeric_value) or numeric_value < 0.0:
            raise ValueError("personalization values must be finite and non-negative")
        if node_id in node_set:
            raw[node_id] += numeric_value
    total = sum(raw.values())
    if total == 0.0:
        if not nodes:
            return {}
        uniform = 1.0 / len(nodes)
        return {node: uniform for node in nodes}
    return {node: raw.get(node, 0.0) / total for node in nodes}


def _pagerank_distribution(
    edges: Iterable[object],
    personalization: Mapping[object, float] | Iterable[object] | object,
    *,
    nodes: Iterable[object] | None,
    task_nodes: Iterable[object] | None,
    restart: float,
    max_iterations: int,
    tolerance: float,
    require_convergence: bool,
) -> dict[NodeId, float]:
    if not isfinite(float(restart)) or not 0.0 <= restart <= 1.0:
        raise ValueError("restart must be between 0 and 1")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if not isfinite(float(tolerance)) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    requested_nodes = {
        _node_value(value)
        for value in (nodes or ())
        if _node_value(value)
    }
    if isinstance(personalization, Mapping):
        requested_nodes.update(
            _node_value(value) for value in personalization if _node_value(value)
        )
    elif isinstance(personalization, (str, GraphNode)):
        node_id = _node_value(personalization)
        if node_id:
            requested_nodes.add(node_id)
    else:
        try:
            personalization = tuple(personalization)  # type: ignore[arg-type]
            requested_nodes.update(
                _node_value(value) for value in personalization if _node_value(value)
            )
        except TypeError:
            node_id = _node_value(personalization)
            if node_id:
                requested_nodes.add(node_id)
    task_set = (
        {_node_value(value) for value in task_nodes if _node_value(value)}
        if task_nodes is not None
        else None
    )
    edge_weights: dict[tuple[NodeId, NodeId, bool], list[float]] = defaultdict(list)
    graph_nodes = set(requested_nodes)
    for edge in edges:
        source, target, weight, directed = _edge_values(edge)
        if task_set is not None and (source not in task_set or target not in task_set):
            continue
        graph_nodes.update((source, target))
        if weight > 0.0:
            if not directed and target < source:
                source, target = target, source
            edge_weights[(source, target, directed)].append(weight)
    if task_set is not None:
        graph_nodes.update(task_set)
        graph_nodes.intersection_update(task_set)
    if not graph_nodes:
        return {}
    ordered_nodes = tuple(sorted(graph_nodes))
    allowed = set(ordered_nodes)
    adjacency: dict[NodeId, dict[NodeId, float]] = {
        node: defaultdict(float) for node in ordered_nodes
    }
    for (source, target, directed), weights in sorted(edge_weights.items()):
        if source not in allowed or target not in allowed:
            continue
        weight = fsum(sorted(weights))
        adjacency[source][target] += weight
        if not directed and source != target:
            adjacency[target][source] += weight

    teleport = _normalize_personalization(personalization, ordered_nodes)
    ranks = dict(teleport)
    walk_probability = 1.0 - float(restart)
    converged = False
    last_delta = float("inf")
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        incoming = {node: 0.0 for node in ordered_nodes}
        dangling = 0.0
        for source in ordered_nodes:
            outgoing = adjacency[source]
            total_weight = fsum(outgoing[target] for target in sorted(outgoing))
            if total_weight == 0.0:
                dangling += ranks[source]
                continue
            for target in sorted(outgoing):
                incoming[target] += ranks[source] * outgoing[target] / total_weight
        next_ranks = {
            node: float(restart) * teleport[node]
            + walk_probability * (incoming[node] + dangling * teleport[node])
            for node in ordered_nodes
        }
        last_delta = sum(abs(next_ranks[node] - ranks[node]) for node in ordered_nodes)
        ranks = next_ranks
        iterations = iteration
        if last_delta <= tolerance:
            converged = True
            break
    if require_convergence and not converged:
        raise PageRankConvergenceError(
            iterations=iterations,
            max_iterations=max_iterations,
            tolerance=float(tolerance),
            last_delta=last_delta,
        )
    return ranks


def weighted_personalized_pagerank(
    edges: Iterable[object],
    personalization: Mapping[object, float] | Iterable[object] | object,
    *,
    nodes: Iterable[object] | None = None,
    task_nodes: Iterable[object] | None = None,
    restart: float = 0.15,
    top_k: int | None = 8,
    max_iterations: int = 100,
    tolerance: float = 1e-12,
) -> tuple[Score, ...]:
    """Run weighted personalized PageRank and return stable top-k scores.

    ``task_nodes`` projects the graph before diffusion, so unrelated edges and
    nodes cannot affect the result.  Ties are broken by node id.  Dangling
    mass is teleported according to the personalization distribution.
    """

    if top_k is not None and (not isinstance(top_k, int) or top_k < 0):
        raise ValueError("top_k must be None or a non-negative integer")
    ranks = _pagerank_distribution(
        edges,
        personalization,
        nodes=nodes,
        task_nodes=task_nodes,
        restart=restart,
        max_iterations=max_iterations,
        tolerance=tolerance,
        require_convergence=False,
    )
    ranked = sorted(ranks.items(), key=lambda item: (-item[1], item[0]))
    return tuple(ranked if top_k is None else ranked[:top_k])


personalized_pagerank = weighted_personalized_pagerank


def pagerank_scores(
    edges: Iterable[object],
    personalization: Mapping[object, float] | Iterable[object] | object,
    *,
    nodes: Iterable[object] | None = None,
    task_nodes: Iterable[object] | None = None,
    restart: float = 0.15,
    max_iterations: int = 100,
    tolerance: float = 1e-12,
) -> dict[NodeId, float]:
    """Return the complete PPR distribution in stable node order.

    Raises :class:`PageRankConvergenceError` when the iteration budget is
    exhausted before the requested tolerance is reached.
    """

    return _pagerank_distribution(
        edges,
        personalization,
        nodes=nodes,
        task_nodes=task_nodes,
        restart=restart,
        max_iterations=max_iterations,
        tolerance=tolerance,
        require_convergence=True,
    )


def _score_map(scores: Mapping[object, float]) -> dict[NodeId, float]:
    normalized: dict[NodeId, float] = {}
    for node, value in scores.items():
        node_id = _node_value(node)
        numeric_value = float(value)
        if not node_id:
            continue
        if not isfinite(numeric_value):
            raise ValueError("scores must be finite")
        normalized[node_id] = numeric_value
    return dict(sorted(normalized.items()))


def combine_scores(
    vector_scores: Mapping[object, float],
    graph_scores: Mapping[object, float],
    *,
    strategy: str = "simple",
    vector_weight: float = 0.5,
    graph_weight: float = 0.5,
) -> dict[NodeId, float]:
    """Select ``simple``, ``graph`` or weighted ``fusion`` scores.

    ``simple`` returns vector scores, ``graph`` returns graph scores, and
    ``fusion`` computes ``vector_weight * vector + graph_weight * graph``.
    Missing scores in fusion are zero.  The returned mapping is inserted in
    node-id order for reproducible serialization.
    """

    selected = _text(strategy).casefold()
    if selected not in {"simple", "graph", "fusion"}:
        raise ValueError("strategy must be one of: simple, graph, fusion")
    vectors = _score_map(vector_scores)
    graphs = _score_map(graph_scores)
    if selected == "simple":
        return vectors
    if selected == "graph":
        return graphs
    vector_factor = float(vector_weight)
    graph_factor = float(graph_weight)
    if (
        not isfinite(vector_factor)
        or not isfinite(graph_factor)
        or vector_factor < 0.0
        or graph_factor < 0.0
        or vector_factor + graph_factor == 0.0
    ):
        raise ValueError("fusion weights must be finite, non-negative, and not both zero")
    nodes = sorted(set(vectors) | set(graphs))
    return {
        node: vector_factor * vectors.get(node, 0.0)
        + graph_factor * graphs.get(node, 0.0)
        for node in nodes
    }


def fuse_scores(
    vector_scores: Mapping[object, float],
    graph_scores: Mapping[object, float],
    *,
    vector_weight: float = 0.5,
    graph_weight: float = 0.5,
) -> dict[NodeId, float]:
    """Return the explicit vector+graph fusion path."""

    return combine_scores(
        vector_scores,
        graph_scores,
        strategy="fusion",
        vector_weight=vector_weight,
        graph_weight=graph_weight,
    )


score_fusion = fuse_scores


def rank_scores(
    scores: Mapping[object, float],
    *,
    top_k: int | None = None,
) -> tuple[Score, ...]:
    """Return scores ranked descending with deterministic id tie-breaking."""

    if top_k is not None and (not isinstance(top_k, int) or top_k < 0):
        raise ValueError("top_k must be None or a non-negative integer")
    ranked = sorted(_score_map(scores).items(), key=lambda item: (-item[1], item[0]))
    return tuple(ranked if top_k is None else ranked[:top_k])


__all__ = [
    "Edge",
    "GraphEdge",
    "GraphNode",
    "Node",
    "PageRankConvergenceError",
    "aggregate_cv_overlap",
    "aggregate_implicit_edge_weights",
    "aggregate_overlap_edges",
    "aggregate_overlap_weights",
    "aggregate_staff_overlap",
    "combine_scores",
    "compute_tag_idf",
    "compute_tag_similarities",
    "count_distinct_author_hashes",
    "count_distinct_authors",
    "cv_overlap_edges",
    "fuse_scores",
    "implicit_edge_weight",
    "pairwise_tag_similarity",
    "pagerank_scores",
    "personalized_pagerank",
    "rank_scores",
    "score_fusion",
    "staff_overlap_edges",
    "subject_tag_similarities",
    "tag_idf_weights",
    "tag_similarity",
    "tag_similarity_edges",
    "weighted_personalized_pagerank",
    "weighted_tag_similarity",
]

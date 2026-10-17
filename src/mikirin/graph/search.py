"""Offline queries over the materialized A5 graph results.

The graph build is intentionally not part of this module.  ``derived_edge``,
``node_pagerank`` and ``task_pagerank`` are read-only inputs; this module only selects rows and joins
them with candidates supplied by a caller.  In particular, ``simple`` is the
default mode and does not consult either graph table.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from mikirin.graph.tasks import HETEROGENEOUS_TASKS, TASK_GRAPH_VERSION


SearchMode = Literal["simple", "graph", "fusion"]

DEFAULT_MODE: SearchMode = "simple"
DEFAULT_RESTART = 0.15
DEFAULT_TOP_K = 8
DEFAULT_FUSION_WEIGHTS = (0.45, 0.25, 0.30)
VALID_MODES = frozenset(("simple", "graph", "fusion"))
TASK_EDGE_KINDS = {
    "similar_work": ("tag_similarity", "same_season"),
    **HETEROGENEOUS_TASKS,
}


@dataclass(frozen=True, slots=True)
class GraphTask:
    """The materialized graph slice used by one search task.

    ``graph_kind`` names the corresponding materialized task.  The
    optional ``edge_kinds`` narrows ``derived_edge.kind`` without changing the
    database schema.  A task is only a query-time boundary; no online graph
    expansion is performed here.

    ``voice_cast`` is person/character/subject and ``staff_filmography`` is
    person/subject. Those tasks require versioned heterogeneous task_pagerank
    rows; they never fall back to old subject-overlap projections.
    """

    name: str
    graph_kind: str | None = None
    edge_kinds: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One result with all score channels kept separate for evaluation."""

    entity_id: str
    kind: str
    raw_score: float | None
    graph_score: float | None
    fusion_score: float | None
    provenance: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind,
            "raw_score": self.raw_score,
            "graph_score": self.graph_score,
            "fusion_score": self.fusion_score,
            "provenance": [dict(item) for item in self.provenance],
        }


@dataclass(frozen=True, slots=True)
class ABQuery:
    """An anonymous, unrated query record for a blind A/B package."""

    query_id: str
    query: str
    seed_entity_ids: tuple[str, ...] = ()
    task: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "seed_entity_ids": list(self.seed_entity_ids),
            "task": self.task,
        }


@dataclass(frozen=True, slots=True)
class ABExperimentPackage:
    """The data-only part of a 30-query A/B experiment.

    There is deliberately no ratings, preference, user id, or generated
    outcome field.  A later human-review tool can attach those outside this
    package after anonymous presentation.
    """

    experiment_id: str
    arm_a: SearchMode
    arm_b: SearchMode
    queries: tuple[ABQuery, ...]
    version: str = "a5-ab-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "experiment_id": self.experiment_id,
            "arms": {"a": self.arm_a, "b": self.arm_b},
            "queries": [query.to_dict() for query in self.queries],
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    entity_id: str
    source: str
    kind: str
    raw_score: float | None
    ordinal: int


@dataclass(slots=True)
class _GraphNeighbor:
    entity_id: str
    edge_weight: float
    graph_score: float | None
    edge_kinds: set[str]
    provenance: list[Mapping[str, Any]]
    entity_type: str | None = None


class OfflineGraphSearch:
    """Read-only search facade for materialized A5 graph data.

    Candidate inputs are intentionally injected.  A vector index, FTS query,
    or ``SemanticEpisodeSearch`` object may be used by the caller, but this
    class does not invoke any of them and therefore remains offline and
    deterministic.
    """

    def __init__(self, database: Path | str | sqlite3.Connection) -> None:
        self.database = database

    def search(
        self,
        seed_entity_ids: Iterable[str] | str = (),
        *,
        mode: SearchMode = DEFAULT_MODE,
        task: GraphTask | str | None = None,
        graph_kind: str | None = None,
        edge_kinds: Iterable[str] | str | None = None,
        build_id: str | None = None,
        vector_candidates: Iterable[Any] | None = (),
        lexical_candidates: Iterable[Any] | None = (),
        top_k: int = DEFAULT_TOP_K,
        limit: int | None = None,
        restart: float = DEFAULT_RESTART,
        fusion_weights: Sequence[float] = DEFAULT_FUSION_WEIGHTS,
        include_seeds: bool = False,
    ) -> list[SearchResult]:
        """Return at most ``top_k`` deterministic results.

        ``mode`` is always explicit in the returned behavior: the default is
        ``simple``; ``graph`` and ``fusion`` must be requested by name.  The
        ``restart`` value documents the PPR contract (0.15 by default) and is
        checked against heterogeneous graph provenance. Seed PPR is materialized
        into ``task_pagerank``. Credit tasks reject requests above the built
        top-k capacity; requesting fewer results is safe.
        """

        selected_mode = _validate_mode(mode)
        top_k = _validate_top_k(limit if limit is not None else top_k)
        restart = _validate_restart(restart)
        weights = _validate_fusion_weights(fusion_weights)
        seeds = _normalize_ids(seed_entity_ids)
        vector = _normalize_candidates(vector_candidates, "vector")
        lexical = _normalize_candidates(lexical_candidates, "lexical")

        if selected_mode == "simple":
            return _simple_results(vector, lexical, top_k)
        if build_id is None:
            raise ValueError("graph and fusion modes require an explicit build_id")

        resolved_task = _resolve_task(task, graph_kind, edge_kinds)
        if resolved_task.graph_kind is None:
            raise ValueError("graph and fusion modes require a task graph_kind")
        with self._read_connection() as db:
            if resolved_task.graph_kind not in HETEROGENEOUS_TASKS:
                _require_materialized_tables(db)
            neighbors = _load_neighbors(
                db,
                seeds,
                graph_kind=resolved_task.graph_kind,
                edge_kinds=resolved_task.edge_kinds,
                build_id=build_id,
                include_seeds=include_seeds,
                restart=restart,
                top_k=top_k,
            )

            if selected_mode == "graph":
                return _graph_results(neighbors, top_k)
            return _fusion_results(
                vector,
                lexical,
                neighbors,
                top_k,
                weights=weights,
            )

    def query_task_subgraph(self, *args: Any, **kwargs: Any) -> list[SearchResult]:
        """Named alias for callers that want to emphasize the task boundary."""

        return self.search(*args, **kwargs)

    def search_task_subgraph(self, *args: Any, **kwargs: Any) -> list[SearchResult]:
        """Compatibility alias for the task-subgraph query operation."""

        return self.search(*args, **kwargs)

    @contextmanager
    def _read_connection(self) -> Iterator[sqlite3.Connection]:
        if isinstance(self.database, sqlite3.Connection):
            yield self.database
            return

        database = str(self.database)
        if database == ":memory:":
            with closing(sqlite3.connect(database)) as db:
                yield db
            return

        path = Path(self.database).resolve()
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            yield db


GraphSearch = OfflineGraphSearch
MaterializedGraphSearch = OfflineGraphSearch
GraphSearchResult = SearchResult
ABPackage = ABExperimentPackage


def search(
    database: Path | str | sqlite3.Connection,
    seed_entity_ids: Iterable[str] | str = (),
    **kwargs: Any,
) -> list[SearchResult]:
    """Functional convenience wrapper around :class:`OfflineGraphSearch`."""

    return OfflineGraphSearch(database).search(seed_entity_ids, **kwargs)


def build_ab_package(
    queries: Iterable[str | Mapping[str, Any]],
    *,
    arm_a: SearchMode = "simple",
    arm_b: SearchMode = "fusion",
    experiment_id: str | None = None,
) -> ABExperimentPackage:
    """Build exactly 30 anonymous query records without fabricating ratings."""

    arm_a = _validate_mode(arm_a)
    arm_b = _validate_mode(arm_b)
    if arm_a == arm_b:
        raise ValueError("A/B arms must use two different search modes")

    specs = list(queries)
    if len(specs) != 30:
        raise ValueError("A5 A/B packages require exactly 30 queries")

    normalized: list[dict[str, Any]] = []
    for item in specs:
        if isinstance(item, str):
            query = item.strip()
            seeds: tuple[str, ...] = ()
            task = None
        elif isinstance(item, Mapping):
            query = str(item.get("query", "")).strip()
            seeds = _normalize_ids(item.get("seed_entity_ids", ()))
            raw_task = item.get("task")
            task = None if raw_task is None else str(raw_task)
        else:
            raise TypeError("A/B queries must be strings or mappings")
        if not query:
            raise ValueError("A/B queries must contain non-empty query text")
        normalized.append({"query": query, "seed_entity_ids": seeds, "task": task})

    canonical = json.dumps(
        {
            "arms": [arm_a, arm_b],
            "queries": [
                {
                    "query": item["query"],
                    "seed_entity_ids": list(item["seed_entity_ids"]),
                    "task": item["task"],
                }
                for item in normalized
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    package_id = experiment_id or f"a5-{digest[:16]}"
    records = tuple(
        ABQuery(
            query_id=(
                f"q-{index + 1:02d}-"
                f"{hashlib.sha256((digest + str(index)).encode()).hexdigest()[:10]}"
            ),
            query=item["query"],
            seed_entity_ids=item["seed_entity_ids"],
            task=item["task"],
        )
        for index, item in enumerate(normalized)
    )
    return ABExperimentPackage(package_id, arm_a, arm_b, records)


make_ab_package = build_ab_package


def _validate_mode(mode: str) -> SearchMode:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)!r}, got {mode!r}")
    return mode  # type: ignore[return-value]


def _validate_top_k(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("top_k must be a positive integer")
    return value


def _validate_restart(value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("restart must be a finite value between 0 and 1") from error
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("restart must be a finite value between 0 and 1")
    return result


def _validate_fusion_weights(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError("fusion_weights must contain vector, lexical, and graph weights")
    try:
        weights = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError("fusion_weights must be finite non-negative numbers") from error
    if any(not math.isfinite(item) or item < 0 for item in weights) or not any(weights):
        raise ValueError("fusion_weights must be finite non-negative numbers")
    return weights  # type: ignore[return-value]


def _normalize_ids(values: Iterable[str] | str | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return tuple(result)


def _resolve_task(
    task: GraphTask | str | None,
    graph_kind: str | None,
    edge_kinds: Iterable[str] | str | None,
) -> GraphTask:
    if isinstance(task, GraphTask):
        task_graph_kind = graph_kind if graph_kind is not None else task.graph_kind
        task_edge_kinds: Iterable[str] | str = (
            edge_kinds if edge_kinds is not None else task.edge_kinds
        )
        name = task.name
    else:
        name = "default" if task is None else str(task)
        task_graph_kind = graph_kind if graph_kind is not None else (str(task) if task else None)
        task_edge_kinds = edge_kinds or ()
    if not task_edge_kinds:
        task_edge_kinds = TASK_EDGE_KINDS.get(task_graph_kind, ())
    if isinstance(task_edge_kinds, str):
        kinds = (task_edge_kinds,)
    else:
        kinds = tuple(sorted({str(item) for item in task_edge_kinds if str(item)}))
    return GraphTask(name=name, graph_kind=task_graph_kind, edge_kinds=kinds)


def _normalize_candidates(values: Iterable[Any] | None, source: str) -> list[_Candidate]:
    if values is None:
        return []
    candidates: list[_Candidate] = []
    for ordinal, item in enumerate(values):
        entity_id, kind, raw_score = _candidate_fields(item, source)
        if entity_id is None:
            continue
        candidates.append(_Candidate(entity_id, source, kind, raw_score, ordinal))

    by_id: dict[str, _Candidate] = {}
    for candidate in candidates:
        existing = by_id.get(candidate.entity_id)
        if existing is None or _candidate_is_better(candidate, existing):
            by_id[candidate.entity_id] = candidate
    return sorted(
        by_id.values(),
        key=lambda item: (
            item.raw_score is None,
            -(item.raw_score or 0.0),
            item.ordinal,
            item.entity_id,
        ),
    )


def _candidate_fields(item: Any, source: str) -> tuple[str | None, str, float | None]:
    if isinstance(item, (tuple, list)):
        if len(item) >= 3:
            kind, entity_id, score = item[0], item[1], item[2]
        elif len(item) == 2:
            entity_id, score = item
            kind = source
        elif len(item) == 1:
            entity_id, kind, score = item[0], source, None
        else:
            return None, source, None
    else:
        entity_id = _field(
            item,
            "entity_id",
            "episode_entity_id",
            "subject_entity_id",
            "node_entity_id",
            "id",
        )
        kind = _field(item, "kind", "object_kind", "source_kind") or source
        score = _field(item, "raw_score", "score", "lexical_score")

    entity_id = None if entity_id is None else str(entity_id).strip()
    if not entity_id:
        return None, str(kind), _finite_float(score)
    return entity_id, str(kind), _finite_float(score)


def _field(item: Any, *names: str) -> Any:
    if isinstance(item, Mapping):
        for name in names:
            if name in item:
                return item[name]
        return None
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _candidate_is_better(left: _Candidate, right: _Candidate) -> bool:
    if left.raw_score is None:
        return False
    if right.raw_score is None:
        return True
    if left.raw_score != right.raw_score:
        return left.raw_score > right.raw_score
    return (left.ordinal, left.kind) < (right.ordinal, right.kind)


def _simple_results(
    vector: list[_Candidate], lexical: list[_Candidate], top_k: int
) -> list[SearchResult]:
    merged = _merge_candidates(vector, lexical)
    ordered = sorted(
        merged.values(),
        key=lambda item: (
            item["raw_score"] is None,
            -(item["raw_score"] or 0.0),
            item["kind"],
            item["entity_id"],
        ),
    )
    return [
        SearchResult(
            entity_id=item["entity_id"],
            kind=item["kind"],
            raw_score=item["raw_score"],
            graph_score=None,
            fusion_score=None,
            provenance=tuple(item["provenance"]),
        )
        for item in ordered[:top_k]
    ]


def _merge_candidates(
    vector: list[_Candidate], lexical: list[_Candidate]
) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for candidate in (*vector, *lexical):
        item = merged.setdefault(
            candidate.entity_id,
            {
                "entity_id": candidate.entity_id,
                "kind": candidate.kind,
                "raw_score": candidate.raw_score,
                "candidates": {},
                "provenance": [],
            },
        )
        if candidate.source not in item["candidates"] or _candidate_is_better(
            candidate, item["candidates"][candidate.source]
        ):
            item["candidates"][candidate.source] = candidate
        if candidate.source == "vector" or item["kind"] in {"vector", "lexical"}:
            item["kind"] = candidate.kind
        if candidate.raw_score is not None and (
            item["raw_score"] is None or candidate.raw_score > item["raw_score"]
        ):
            item["raw_score"] = candidate.raw_score
        item["provenance"].append(
            {
                "source": candidate.source,
                "kind": candidate.kind,
                "raw_score": candidate.raw_score,
                "rank": candidate.ordinal + 1,
            }
        )
    for item in merged.values():
        item["provenance"] = _dedupe_provenance(item["provenance"])
    return merged


def _graph_results(neighbors: list[_GraphNeighbor], top_k: int) -> list[SearchResult]:
    ordered = sorted(
        neighbors,
        key=lambda item: (
            -(item.graph_score if item.graph_score is not None else 0.0),
            -item.edge_weight,
            item.entity_id,
        ),
    )
    return [
        SearchResult(
            entity_id=item.entity_id,
            kind=_graph_kind(item),
            raw_score=item.edge_weight,
            graph_score=item.graph_score,
            fusion_score=None,
            provenance=tuple(item.provenance),
        )
        for item in ordered[:top_k]
    ]


def _fusion_results(
    vector: list[_Candidate],
    lexical: list[_Candidate],
    neighbors: list[_GraphNeighbor],
    top_k: int,
    *,
    weights: tuple[float, float, float],
) -> list[SearchResult]:
    merged = _merge_candidates(vector, lexical)
    graph_by_id = {item.entity_id: item for item in neighbors}
    graph_ordered = sorted(
        neighbors,
        key=lambda item: (
            -(item.graph_score if item.graph_score is not None else 0.0),
            -item.edge_weight,
            item.entity_id,
        ),
    )
    # PPR top-k is an explicit bound on the graph contribution to fusion.
    graph_by_id = {item.entity_id: item for item in graph_ordered[:top_k]}

    vector_rank = {item.entity_id: index for index, item in enumerate(vector)}
    lexical_rank = {item.entity_id: index for index, item in enumerate(lexical)}
    graph_rank = {item.entity_id: index for index, item in enumerate(graph_ordered[:top_k])}
    for neighbor in graph_by_id.values():
        item = merged.setdefault(
            neighbor.entity_id,
            {
                "entity_id": neighbor.entity_id,
                "kind": _graph_kind(neighbor),
                "raw_score": neighbor.edge_weight,
                "candidates": {},
                "provenance": [],
            },
        )
        if item["raw_score"] is None or neighbor.edge_weight > item["raw_score"]:
            item["raw_score"] = neighbor.edge_weight
        item["graph"] = neighbor
        item["provenance"].extend(neighbor.provenance)

    vector_count = max(len(vector), 1)
    lexical_count = max(len(lexical), 1)
    graph_count = max(len(graph_rank), 1)
    results: list[SearchResult] = []
    for item in merged.values():
        entity_id = item["entity_id"]
        vector_component = (
            (len(vector) - vector_rank[entity_id]) / vector_count
            if entity_id in vector_rank
            else 0.0
        )
        lexical_component = (
            (len(lexical) - lexical_rank[entity_id]) / lexical_count
            if entity_id in lexical_rank
            else 0.0
        )
        graph_component = (
            (len(graph_rank) - graph_rank[entity_id]) / graph_count
            if entity_id in graph_rank
            else 0.0
        )
        fusion_score = (
            weights[0] * vector_component
            + weights[1] * lexical_component
            + weights[2] * graph_component
        )
        graph = item.get("graph")
        graph_score = graph.graph_score if graph is not None else None
        item["provenance"] = _dedupe_provenance(item["provenance"])
        provenance = list(item["provenance"])
        provenance.append(
            {
                "source": "fusion",
                "vector_component": vector_component,
                "lexical_component": lexical_component,
                "graph_component": graph_component,
                "weights": {
                    "vector": weights[0],
                    "lexical": weights[1],
                    "graph": weights[2],
                },
            }
        )
        results.append(
            SearchResult(
                entity_id=entity_id,
                kind=item["kind"],
                raw_score=item["raw_score"],
                graph_score=graph_score,
                fusion_score=fusion_score,
                provenance=tuple(provenance),
            )
        )
    results.sort(
        key=lambda item: (
            -(item.fusion_score or 0.0),
            -(item.graph_score or 0.0),
            -(item.raw_score or 0.0),
            item.entity_id,
            item.kind,
        )
    )
    return results[:top_k]


def _require_materialized_tables(db: sqlite3.Connection) -> None:
    names = {
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "('derived_edge', 'node_pagerank')"
        )
    }
    missing = {"derived_edge", "node_pagerank"} - names
    if missing:
        raise RuntimeError(
            "materialized A5 tables missing: " + ", ".join(sorted(missing))
        )


def _load_neighbors(
    db: sqlite3.Connection,
    seeds: tuple[str, ...],
    *,
    graph_kind: str | None,
    edge_kinds: tuple[str, ...],
    build_id: str | None,
    include_seeds: bool,
    restart: float,
    top_k: int,
) -> list[_GraphNeighbor]:
    if not seeds:
        return []
    if graph_kind in HETEROGENEOUS_TASKS:
        return _load_credit_neighbors(
            db, seeds, graph_kind=graph_kind, edge_kinds=edge_kinds,
            build_id=build_id, include_seeds=include_seeds, restart=restart, top_k=top_k,
        )

    seed_marks = ",".join("?" for _ in seeds)
    predicates = [f"(src_entity_id IN ({seed_marks}) OR dst_entity_id IN ({seed_marks}))"]
    parameters: list[Any] = [*seeds, *seeds]
    if build_id is not None:
        predicates.append("build_id=?")
        parameters.append(build_id)
    if edge_kinds:
        marks = ",".join("?" for _ in edge_kinds)
        predicates.append(f"kind IN ({marks})")
        parameters.extend(edge_kinds)
    rows = db.execute(
        "SELECT src_entity_id, dst_entity_id, kind, weight, evidence_json, build_id "
        "FROM derived_edge WHERE " + " AND ".join(predicates) + " "
        "ORDER BY weight DESC, src_entity_id, dst_entity_id, kind, build_id",
        parameters,
    ).fetchall()

    neighbors: dict[str, _GraphNeighbor] = {}
    for src, dst, kind, weight, evidence_json, row_build_id in rows:
        src = str(src)
        dst = str(dst)
        neighbor_id = dst if src in seeds else src
        if not include_seeds and neighbor_id in seeds:
            continue
        weight = float(weight)
        item = neighbors.get(neighbor_id)
        if item is None:
            item = _GraphNeighbor(
                entity_id=neighbor_id,
                edge_weight=weight,
                graph_score=None,
                edge_kinds=set(),
                provenance=[],
            )
            neighbors[neighbor_id] = item
        item.edge_weight = max(item.edge_weight, weight)
        item.edge_kinds.add(str(kind))
        evidence = _json_value(evidence_json)
        item.provenance.append(
            {
                "source": "derived_edge",
                "kind": str(kind),
                "seed_entity_ids": [seed for seed in seeds if seed in (src, dst)],
                "weight": weight,
                "build_id": str(row_build_id),
                "evidence": evidence,
            }
        )

    task_table_exists = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_pagerank'"
    ).fetchone()
    if task_table_exists and graph_kind is not None and build_id is not None:
        task_rows = db.execute(
            f"SELECT seed_entity_id,node_entity_id,rank,score,evidence_json "
            f"FROM task_pagerank WHERE seed_entity_id IN ({seed_marks}) "
            "AND graph_kind=? AND build_id=? ORDER BY seed_entity_id,rank,node_entity_id",
            (*seeds, graph_kind, build_id),
        ).fetchall()
        personalized_nodes: set[str] = set()
        for seed, node_id, rank, score, evidence_json in task_rows:
            evidence = _json_value(evidence_json)
            if isinstance(evidence, Mapping):
                stored_restart = evidence.get("restart", DEFAULT_RESTART)
                if not math.isclose(float(stored_restart), restart, abs_tol=1e-12):
                    raise ValueError("requested restart differs from materialized task PageRank")
                stored_kinds = evidence.get("edge_kinds")
                if stored_kinds and edge_kinds and set(stored_kinds) != set(edge_kinds):
                    raise ValueError("requested edge kinds differ from materialized task PageRank")
            node_id = str(node_id)
            if not include_seeds and node_id in seeds:
                continue
            personalized_nodes.add(node_id)
            item = neighbors.get(node_id)
            if item is None:
                item = _GraphNeighbor(
                    entity_id=node_id,
                    edge_weight=0.0,
                    graph_score=float(score),
                    edge_kinds=set(edge_kinds),
                    provenance=[],
                )
                neighbors[node_id] = item
            elif item.graph_score is None or float(score) > item.graph_score:
                item.graph_score = float(score)
            item.provenance.append(
                {
                    "source": "task_pagerank",
                    "seed_entity_id": str(seed),
                    "graph_kind": graph_kind,
                    "rank": int(rank),
                    "score": float(score),
                    "build_id": build_id,
                    "evidence": _json_value(evidence_json),
                }
            )
        if task_rows:
            # Global rank and seed PPR have different normalizations. Never mix
            # non-personalized neighbors into a materialized seed distribution.
            return [neighbors[node] for node in sorted(personalized_nodes)]

    if not neighbors:
        return []

    node_ids = tuple(sorted(neighbors))
    node_marks = ",".join("?" for _ in node_ids)
    node_predicates = [f"node_entity_id IN ({node_marks})"]
    node_parameters: list[Any] = [*node_ids]
    if graph_kind is not None:
        node_predicates.append("graph_kind=?")
        node_parameters.append(graph_kind)
    if build_id is not None:
        node_predicates.append("build_id=?")
        node_parameters.append(build_id)
    pagerank_rows = db.execute(
        "SELECT node_entity_id, graph_kind, score, build_id FROM node_pagerank WHERE "
        + " AND ".join(node_predicates)
        + " ORDER BY score DESC, node_entity_id, graph_kind, build_id",
        node_parameters,
    ).fetchall()
    for node_id, row_graph_kind, score, row_build_id in pagerank_rows:
        item = neighbors.get(str(node_id))
        if item is None:
            continue
        score = float(score)
        if item.graph_score is None:
            item.graph_score = score
        item.provenance.append(
            {
                "source": "node_pagerank",
                "graph_kind": str(row_graph_kind),
                "score": score,
                "build_id": str(row_build_id),
                "ppr_restart": restart,
                "ppr_top_k": top_k,
            }
        )
    return list(neighbors.values())


def _graph_kind(item: _GraphNeighbor) -> str:
    if item.entity_type is not None:
        return item.entity_type
    if len(item.edge_kinds) == 1:
        return next(iter(item.edge_kinds))
    return "derived_edge"


def _load_credit_neighbors(
    db: sqlite3.Connection, seeds: tuple[str, ...], *, graph_kind: str,
    edge_kinds: tuple[str, ...], build_id: str | None, include_seeds: bool,
    restart: float, top_k: int,
) -> list[_GraphNeighbor]:
    """Only table lookups: never read source relations or reconstruct a graph."""
    if set(edge_kinds) != set(HETEROGENEOUS_TASKS[graph_kind]):
        raise ValueError("requested edge kinds differ from the materialized credit task")
    if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_pagerank'").fetchone():
        raise RuntimeError("task_pagerank is missing; build the heterogeneous task graphs offline")
    marks = ",".join("?" for _ in seeds)
    rows = db.execute(
        f"SELECT seed_entity_id,node_entity_id,rank,score,evidence_json FROM task_pagerank "
        f"WHERE seed_entity_id IN ({marks}) AND graph_kind=? AND build_id=? "
        "ORDER BY seed_entity_id,rank,node_entity_id", (*seeds, graph_kind, build_id),
    ).fetchall()
    neighbors: dict[str, _GraphNeighbor] = {}
    for seed, node, rank, score, raw in rows:
        evidence = _json_value(raw)
        if not isinstance(evidence, Mapping) or evidence.get("graph_version") != TASK_GRAPH_VERSION:
            raise ValueError("legacy overlap projection is not a credit graph; rebuild task_pagerank offline")
        stored_restart = _finite_float(evidence.get("restart"))
        if stored_restart is None or not math.isclose(stored_restart, restart, rel_tol=0, abs_tol=1e-12):
            raise ValueError("requested restart differs from materialized task PageRank")
        if set(evidence.get("edge_kinds", ())) != set(edge_kinds):
            raise ValueError("requested edge kinds differ from materialized task PageRank")
        stored_top_k = evidence.get("top_k")
        if isinstance(stored_top_k, bool) or not isinstance(stored_top_k, int) or top_k > stored_top_k:
            raise ValueError("requested top_k exceeds the materialized task PageRank capacity")
        score = _finite_float(score)
        entity_type = evidence.get("node_entity_type")
        if score is None or score < 0 or entity_type not in {"person", "character", "subject"}:
            raise ValueError("invalid materialized credit result")
        if not include_seeds and node in seeds:
            continue
        item = neighbors.setdefault(node, _GraphNeighbor(node, 0.0, score, set(edge_kinds), [], entity_type))
        item.graph_score = max(item.graph_score, score)
        item.provenance.append({"source": "task_pagerank", "seed_entity_id": seed,
                                "graph_kind": graph_kind, "rank": rank, "score": score,
                                "build_id": build_id, "evidence": dict(evidence)})
    return list(neighbors.values())


def _json_value(value: Any) -> Any:
    if value is None:
        return {}
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


def _dedupe_provenance(items: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = json.dumps(dict(item), ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            result.append(dict(item))
            seen.add(key)
    return result

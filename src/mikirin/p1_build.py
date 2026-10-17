from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Iterator

from mikirin.atomic import atomic_write_json
from mikirin.schema import CONTENT_SCHEMA, CONTENT_SCHEMA_VERSION


P1_PROCESSOR_VERSION = "p1-build-v1"
_P1_TABLES = {
    "slang_run",
    "slang_candidate",
    "slang_candidate_subject_stat",
    "slang_scan_checkpoint",
    "slang_evidence",
    "slang_link_candidate",
    "cooccurrence",
    "derived_edge",
    "node_pagerank",
    "task_pagerank",
    "implicit_edge_evidence",
    "implicit_edge",
}
_P1_INDEX_PREFIXES = (
    "idx_slang_",
    "idx_cooccurrence_",
    "idx_derived_edge_",
    "idx_node_pagerank_",
    "idx_implicit_",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def stable_id(kind: str, *parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{kind}:{digest}"


def _sql_statements(script: str) -> Iterator[str]:
    current: list[str] = []
    for line in script.splitlines():
        current.append(line)
        candidate = "\n".join(current).strip()
        if candidate and sqlite3.complete_statement(candidate):
            yield candidate
            current.clear()
    if any(line.strip() for line in current):
        raise ValueError("incomplete SQL statement in content schema")


def ensure_p1_schema(db: sqlite3.Connection) -> None:
    """Add P1 materialization tables to a cloned v9 database.

    The canonical schema remains the single source of truth: migration statements are
    selected from ``CONTENT_SCHEMA`` and made idempotent instead of being duplicated here.
    """

    for statement in _sql_statements(CONTENT_SCHEMA):
        normalized = " ".join(statement.split())
        upper = normalized.upper()
        if upper.startswith("CREATE TABLE "):
            table = normalized.split()[2]
            if table not in _P1_TABLES:
                continue
            statement = statement.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1)
        elif upper.startswith("CREATE INDEX "):
            index_name = normalized.split()[2]
            if not index_name.startswith(_P1_INDEX_PREFIXES):
                continue
            statement = statement.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS", 1)
        else:
            continue
        db.execute(statement)
    implicit_columns = {
        str(row[1]) for row in db.execute("PRAGMA table_info(implicit_edge_evidence)")
    }
    if "char_start" not in implicit_columns:
        db.execute(
            "ALTER TABLE implicit_edge_evidence ADD COLUMN char_start INTEGER NOT NULL DEFAULT 0"
        )
    if "char_end" not in implicit_columns:
        db.execute(
            "ALTER TABLE implicit_edge_evidence ADD COLUMN char_end INTEGER NOT NULL DEFAULT 0"
        )
    db.execute(
        "INSERT INTO schema_metadata(key,value) VALUES('schema_version',?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(CONTENT_SCHEMA_VERSION),),
    )


@dataclass(frozen=True, slots=True)
class P1BuildConfig:
    anime_subject_type: int = 2
    staff_degree_cap: int = 30
    cv_degree_cap: int = 50
    staff_min_shared: int = 2
    cv_min_shared: int = 2
    tag_df_cap: int = 200
    tag_min_similarity: float = 0.12
    same_season_neighbors: int = 8
    edge_top_k: int = 12
    pagerank_damping: float = 0.85
    pagerank_tolerance: float = 1e-10
    # Real staff/cast graphs need slightly more than 100 iterations to satisfy
    # the strict 1e-10 residual gate.  Keep a deterministic hard ceiling, but
    # leave enough headroom that the full corpus is not rejected at ~1.3e-10.
    pagerank_iterations: int = 200

    def validate(self) -> None:
        if self.staff_degree_cap < 2 or self.cv_degree_cap < 2:
            raise ValueError("overlap degree caps must be at least 2")
        if self.staff_min_shared < 1 or self.cv_min_shared < 1:
            raise ValueError("minimum shared counts must be positive")
        if self.tag_df_cap < 2:
            raise ValueError("tag_df_cap must be at least 2")
        if not 0 <= self.tag_min_similarity <= 1:
            raise ValueError("tag_min_similarity must be in [0, 1]")
        if self.same_season_neighbors < 1 or self.edge_top_k < 1:
            raise ValueError("neighbor limits must be positive")
        if not 0 < self.pagerank_damping < 1:
            raise ValueError("pagerank_damping must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class BuildPaths:
    source: Path
    target: Path
    version: str
    build_id: str


def read_current_database(root: Path) -> tuple[Path, dict[str, Any]]:
    pointer_path = root / "db" / "current.json"
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    database = root / "db" / str(payload["database"])
    if not database.is_file():
        raise FileNotFoundError(f"current database does not exist: {database}")
    return database, payload


def prepare_side_build(root: Path, *, version: str, build_id: str) -> BuildPaths:
    source, pointer = read_current_database(root)
    target = root / "db" / f"content.v{version}.db"
    if target.exists():
        raise FileExistsError(f"side-build target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source, target)
        with closing(sqlite3.connect(target)) as db, db:
            db.execute("PRAGMA foreign_keys=ON")
            ensure_p1_schema(db)
            db.execute(
                "INSERT INTO build(id,started_at,code_version,status,notes) "
                "VALUES(?,?,?,'building',?)",
                (
                    build_id,
                    utc_now(),
                    P1_PROCESSOR_VERSION,
                    json.dumps(
                        {
                            "parent_database": source.name,
                            "parent_version": pointer.get("version"),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                ),
            )
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return BuildPaths(source=source, target=target, version=version, build_id=build_id)


def validate_database(database: Path) -> dict[str, Any]:
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        integrity = str(db.execute("PRAGMA integrity_check").fetchone()[0])
        foreign_keys = db.execute("PRAGMA foreign_key_check").fetchmany(20)
        schema_version = db.execute(
            "SELECT value FROM schema_metadata WHERE key='schema_version'"
        ).fetchone()
        counts = {
            table: int(db.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for table in sorted(_P1_TABLES)
        }
    return {
        "integrity": integrity,
        "foreign_key_errors": [list(row) for row in foreign_keys],
        "schema_version": None if schema_version is None else str(schema_version[0]),
        "counts": counts,
    }


def p1_release_issues(
    database: Path,
    *,
    build_id: str,
    report: dict[str, Any],
) -> list[str]:
    issues: list[str] = []
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        build = db.execute("SELECT status FROM build WHERE id=?", (build_id,)).fetchone()
        if build is None:
            return [f"missing build row: {build_id}"]
        if str(build[0]) not in {"building", "validated"}:
            issues.append(f"build status is not publishable: {build[0]}")
        run_id = str(report.get("stages", {}).get("slang", {}).get("run_id", ""))
        slang = db.execute(
            "SELECT status,threshold_version,input_digest,output_digest FROM slang_run "
            "WHERE id=? AND build_id=?",
            (run_id, build_id),
        ).fetchone()
        if slang is None:
            issues.append("report does not identify a slang run for this build")
        else:
            if str(slang[0]) != "completed" or not slang[2] or not slang[3]:
                issues.append("slang run is incomplete or missing digests")
            if str(slang[1]).startswith("uncalibrated"):
                issues.append("slang thresholds have not been calibrated from user labels")
            if db.execute(
                "SELECT 1 FROM slang_candidate WHERE run_id=? LIMIT 1", (run_id,)
            ).fetchone() is None:
                issues.append("slang run has no shortlisted candidates")
            if db.execute(
                "SELECT 1 FROM cooccurrence WHERE run_id=? LIMIT 1", (run_id,)
            ).fetchone() is None:
                issues.append("slang run has no Pass A cooccurrence rows")
        for kind in ("staff_overlap", "cv_overlap", "tag_similarity", "same_season"):
            if db.execute(
                "SELECT 1 FROM derived_edge WHERE build_id=? AND kind=? LIMIT 1",
                (build_id, kind),
            ).fetchone() is None:
                issues.append(f"derived edge kind is empty: {kind}")
        for graph_kind in ("similar_work", "staff_filmography", "voice_cast"):
            if db.execute(
                "SELECT 1 FROM task_pagerank WHERE build_id=? AND graph_kind=? LIMIT 1",
                (build_id, graph_kind),
            ).fetchone() is None:
                issues.append(f"task PageRank is empty: {graph_kind}")
        implicit_evidence = int(
            db.execute(
                "SELECT count(*) FROM implicit_edge_evidence WHERE run_id=?", (run_id,)
            ).fetchone()[0]
        ) if run_id else 0
        if implicit_evidence < 100:
            issues.append(f"implicit evidence has fewer than 100 reviewable rows: {implicit_evidence}")
    acceptance = report.get("user_acceptance", {})
    for key in (
        "slang_threshold_calibration",
        "golden_query_evaluation",
        "implicit_edge_manual_review",
        "graph_blind_test",
    ):
        if acceptance.get(key) is not True:
            issues.append(f"missing user acceptance: {key}")
    return issues


def publish_side_build(root: Path, paths: BuildPaths, *, report: dict[str, Any]) -> Path:
    validation = validate_database(paths.target)
    if validation["integrity"] != "ok" or validation["foreign_key_errors"]:
        raise RuntimeError(f"P1 database validation failed: {validation}")
    if int(validation["schema_version"] or 0) != CONTENT_SCHEMA_VERSION:
        raise RuntimeError(f"unexpected schema version: {validation['schema_version']}")
    release_issues = p1_release_issues(
        paths.target,
        build_id=paths.build_id,
        report=report,
    )
    if release_issues:
        raise RuntimeError("P1 release gate failed: " + "; ".join(release_issues))
    with closing(sqlite3.connect(paths.target)) as db, db:
        cursor = db.execute(
            "UPDATE build SET completed_at=?,status='published',notes=? WHERE id=?",
            (
                utc_now(),
                json.dumps(report, ensure_ascii=False, sort_keys=True),
                paths.build_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"failed to update build row: {paths.build_id}")
    atomic_write_json(
        root / "db" / "current.json",
        {
            "version": paths.version,
            "database": paths.target.name,
            "published_at": utc_now(),
            "schema_version": CONTENT_SCHEMA_VERSION,
            "parent_database": paths.source.name,
        },
    )
    return paths.target


def _flush_pair_counts(
    db: sqlite3.Connection,
    table: str,
    pairs: dict[tuple[str, str], float],
) -> None:
    if not pairs:
        return
    db.executemany(
        f"INSERT INTO {table}(src,dst,value) VALUES(?,?,?) "
        "ON CONFLICT(src,dst) DO UPDATE SET value=value+excluded.value",
        ((src, dst, value) for (src, dst), value in pairs.items()),
    )
    pairs.clear()


def _materialize_top_k(
    db: sqlite3.Connection,
    *,
    raw_table: str,
    kind: str,
    build_id: str,
    evidence: dict[str, Any],
    top_k: int,
    minimum: float,
) -> int:
    evidence_json = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    before = int(db.execute("SELECT count(*) FROM derived_edge").fetchone()[0])
    db.execute(
        f"""
        WITH directed AS (
            SELECT src AS node, dst AS neighbor, value FROM {raw_table} WHERE value>=?
            UNION ALL
            SELECT dst AS node, src AS neighbor, value FROM {raw_table} WHERE value>=?
        ), ranked AS (
            SELECT node, neighbor, value,
                   row_number() OVER (
                       PARTITION BY node ORDER BY value DESC, neighbor
                   ) AS neighbor_rank
            FROM directed
        ), selected AS (
            SELECT min(node,neighbor) AS src, max(node,neighbor) AS dst
            FROM ranked WHERE neighbor_rank<=? GROUP BY 1,2
        )
        INSERT INTO derived_edge(
            src_entity_id,dst_entity_id,kind,weight,evidence_json,build_id
        )
        SELECT raw.src,raw.dst,?,raw.value,?,?
        FROM {raw_table} raw JOIN selected s ON s.src=raw.src AND s.dst=raw.dst
        WHERE raw.value>=?
        ORDER BY raw.src,raw.dst
        ON CONFLICT(src_entity_id,dst_entity_id,kind,build_id) DO UPDATE SET
            weight=excluded.weight,evidence_json=excluded.evidence_json
        """,
        (minimum, minimum, top_k, kind, evidence_json, build_id, minimum),
    )
    after = int(db.execute("SELECT count(*) FROM derived_edge").fetchone()[0])
    return after - before


def materialize_overlap_edges(
    db: sqlite3.Connection,
    *,
    build_id: str,
    config: P1BuildConfig,
    kind: str,
) -> int:
    if kind == "staff_overlap":
        table, cap, minimum = "rel_subject_person", config.staff_degree_cap, config.staff_min_shared
    elif kind == "cv_overlap":
        table, cap, minimum = "rel_person_character", config.cv_degree_cap, config.cv_min_shared
    else:
        raise ValueError(f"unsupported overlap kind: {kind}")
    raw_table = f"temp.raw_{kind}"
    db.execute(f"DROP TABLE IF EXISTS {raw_table}")
    db.execute(
        f"CREATE TEMP TABLE raw_{kind}(src TEXT NOT NULL,dst TEXT NOT NULL,value REAL NOT NULL,PRIMARY KEY(src,dst)) WITHOUT ROWID"
    )
    rows = db.execute(
        f"""
        SELECT r.person_id,r.subject_id
        FROM {table} r JOIN subjects s ON s.entity_id=r.subject_id
        WHERE s.type=?
        GROUP BY r.person_id,r.subject_id
        ORDER BY r.person_id,r.subject_id
        """,
        (config.anime_subject_type,),
    )
    current_person: str | None = None
    subjects: list[str] = []
    pending: dict[tuple[str, str], float] = defaultdict(float)

    def consume_group() -> None:
        if 2 <= len(subjects) <= cap:
            for src, dst in combinations(subjects, 2):
                pending[(src, dst)] += 1.0
            if len(pending) >= 200_000:
                _flush_pair_counts(db, f"raw_{kind}", pending)

    for person_id, subject_id in rows:
        person_id = str(person_id)
        if current_person is not None and person_id != current_person:
            consume_group()
            subjects = []
        current_person = person_id
        subjects.append(str(subject_id))
    consume_group()
    _flush_pair_counts(db, f"raw_{kind}", pending)
    count = _materialize_top_k(
        db,
        raw_table=f"raw_{kind}",
        kind=kind,
        build_id=build_id,
        evidence={
            "source_table": table,
            "metric": "shared_people_count",
            "person_degree_cap": cap,
            "minimum_shared": minimum,
        },
        top_k=config.edge_top_k,
        minimum=float(minimum),
    )
    db.execute(f"DROP TABLE raw_{kind}")
    return count


def _normalized_tags(raw: str) -> tuple[str, ...]:
    try:
        payload = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return ()
    tags: list[str] = []
    # Use the complete tag payload.  Truncating at 20 silently discards valid
    # low-frequency tags and biases the corpus-wide IDF calculation toward the
    # order chosen by the upstream API.
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("name", "")).strip().casefold()
        if tag and tag not in tags:
            tags.append(tag)
    return tuple(tags)


def materialize_tag_similarity(
    db: sqlite3.Connection, *, build_id: str, config: P1BuildConfig
) -> int:
    subject_tags: dict[str, tuple[str, ...]] = {}
    postings: dict[str, list[str]] = defaultdict(list)
    for entity_id, raw in db.execute(
        "SELECT entity_id,tags_json FROM subjects WHERE type=? ORDER BY entity_id",
        (config.anime_subject_type,),
    ):
        tags = _normalized_tags(str(raw))
        subject_id = str(entity_id)
        subject_tags[subject_id] = tags
        for tag in tags:
            postings[tag].append(subject_id)
    total = max(1, len(subject_tags))
    idf = {
        tag: math.log((total + 1) / (len(ids) + 1)) + 1.0
        for tag, ids in postings.items()
        if 2 <= len(ids) <= config.tag_df_cap
    }
    totals = {
        subject_id: sum(idf.get(tag, 0.0) for tag in tags)
        for subject_id, tags in subject_tags.items()
    }
    db.execute("DROP TABLE IF EXISTS temp.raw_tag_similarity")
    db.execute(
        "CREATE TEMP TABLE raw_tag_similarity(src TEXT NOT NULL,dst TEXT NOT NULL,value REAL NOT NULL,PRIMARY KEY(src,dst)) WITHOUT ROWID"
    )
    common: dict[tuple[str, str], float] = defaultdict(float)
    for tag in sorted(idf):
        weight = idf[tag]
        ids = postings[tag]
        for src, dst in combinations(ids, 2):
            common[(src, dst)] += weight
        if len(common) >= 200_000:
            _flush_pair_counts(db, "raw_tag_similarity", common)
    _flush_pair_counts(db, "raw_tag_similarity", common)
    db.execute("DROP TABLE IF EXISTS temp.tag_subject_weight")
    db.execute(
        "CREATE TEMP TABLE tag_subject_weight(entity_id TEXT PRIMARY KEY,total REAL NOT NULL) WITHOUT ROWID"
    )
    db.executemany(
        "INSERT INTO tag_subject_weight(entity_id,total) VALUES(?,?)",
        sorted(totals.items()),
    )
    db.execute(
        """
        UPDATE raw_tag_similarity AS raw
        SET value=value/(
            (SELECT total FROM tag_subject_weight WHERE entity_id=raw.src)+
            (SELECT total FROM tag_subject_weight WHERE entity_id=raw.dst)-value
        )
        """
    )
    count = _materialize_top_k(
        db,
        raw_table="raw_tag_similarity",
        kind="tag_similarity",
        build_id=build_id,
        evidence={
            "source_column": "subjects.tags_json",
            "metric": "idf_weighted_jaccard",
            "idf_formula": "log((N+1)/(df+1))+1",
            "tag_df_cap": config.tag_df_cap,
        },
        top_k=config.edge_top_k,
        minimum=config.tag_min_similarity,
    )
    db.execute("DROP TABLE raw_tag_similarity")
    db.execute("DROP TABLE tag_subject_weight")
    return count


def _season(value: str | None) -> str | None:
    if not value or len(value) < 7:
        return None
    try:
        year = int(value[:4])
        month = int(value[5:7])
    except ValueError:
        return None
    if not 1 <= month <= 12:
        return None
    return f"{year:04d}-Q{((month - 1) // 3) + 1}"


def materialize_same_season(
    db: sqlite3.Connection, *, build_id: str, config: P1BuildConfig
) -> int:
    rank_build_row = db.execute(
        "SELECT build_id,count(*) AS row_count FROM score_snapshot "
        "GROUP BY build_id ORDER BY row_count DESC,build_id DESC LIMIT 1"
    ).fetchone()
    rank_build_id = None if rank_build_row is None else str(rank_build_row[0])
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    if rank_build_id is None:
        rows = db.execute(
            "SELECT entity_id,date,999999999 FROM subjects WHERE type=? ORDER BY entity_id",
            (config.anime_subject_type,),
        )
    else:
        rows = db.execute(
            """
            SELECT s.entity_id,s.date,coalesce(sc.rank,999999999)
            FROM subjects s LEFT JOIN score_snapshot sc
              ON sc.subject_entity_id=s.entity_id AND sc.build_id=?
            WHERE s.type=? ORDER BY s.entity_id
            """,
            (rank_build_id, config.anime_subject_type),
        )
    for entity_id, date, rank in rows:
        season = _season(None if date is None else str(date))
        if season is not None:
            groups[season].append((int(rank), str(entity_id)))
    db.execute("DROP TABLE IF EXISTS temp.raw_same_season")
    db.execute(
        "CREATE TEMP TABLE raw_same_season(src TEXT NOT NULL,dst TEXT NOT NULL,value REAL NOT NULL,PRIMARY KEY(src,dst)) WITHOUT ROWID"
    )
    edges: dict[tuple[str, str], float] = {}
    radius = max(1, (config.same_season_neighbors + 1) // 2)
    for season in sorted(groups):
        values = sorted(groups[season])
        for index, (_, entity_id) in enumerate(values):
            for other_index in range(index + 1, min(len(values), index + radius + 1)):
                other_id = values[other_index][1]
                src, dst = sorted((entity_id, other_id))
                edges[(src, dst)] = max(edges.get((src, dst), 0.0), 1.0)
        if len(edges) >= 200_000:
            _flush_pair_counts(db, "raw_same_season", edges)
    _flush_pair_counts(db, "raw_same_season", edges)
    count = _materialize_top_k(
        db,
        raw_table="raw_same_season",
        kind="same_season",
        build_id=build_id,
        evidence={
            "source_column": "subjects.date",
            "season_definition": "calendar_quarter",
            "neighbor_order": "score_rank_then_entity_id",
            "score_build_id": rank_build_id,
        },
        top_k=config.edge_top_k,
        minimum=1.0,
    )
    db.execute("DROP TABLE raw_same_season")
    return count


def materialize_derived_edges(
    database: Path, *, build_id: str, config: P1BuildConfig | None = None
) -> dict[str, int]:
    cfg = config or P1BuildConfig()
    cfg.validate()
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-262144")
        ensure_p1_schema(db)
        db.execute("DELETE FROM derived_edge WHERE build_id=?", (build_id,))
        counts = {
            "staff_overlap": materialize_overlap_edges(
                db, build_id=build_id, config=cfg, kind="staff_overlap"
            ),
            "cv_overlap": materialize_overlap_edges(
                db, build_id=build_id, config=cfg, kind="cv_overlap"
            ),
            "tag_similarity": materialize_tag_similarity(db, build_id=build_id, config=cfg),
            "same_season": materialize_same_season(db, build_id=build_id, config=cfg),
        }
    return counts


def materialize_node_pagerank(
    database: Path, *, build_id: str, config: P1BuildConfig | None = None
) -> dict[str, int]:
    from mikirin.graph.algorithms import pagerank_scores

    cfg = config or P1BuildConfig()
    cfg.validate()
    result: dict[str, int] = {}
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        ensure_p1_schema(db)
        if db.execute("SELECT 1 FROM build WHERE id=?", (build_id,)).fetchone() is None:
            raise KeyError(f"unknown build_id: {build_id}")
        edge_count = int(
            db.execute("SELECT count(*) FROM derived_edge WHERE build_id=?", (build_id,)).fetchone()[0]
        )
        if edge_count == 0:
            raise RuntimeError(f"cannot materialize PageRank without derived edges: {build_id}")
        db.execute("DELETE FROM node_pagerank WHERE build_id=?", (build_id,))
        for kind in ("staff_overlap", "cv_overlap", "tag_similarity", "same_season"):
            edges = [
                (str(src), str(dst), float(weight))
                for src, dst, weight in db.execute(
                    "SELECT src_entity_id,dst_entity_id,weight FROM derived_edge "
                    "WHERE build_id=? AND kind=? ORDER BY src_entity_id,dst_entity_id",
                    (build_id, kind),
                )
            ]
            nodes = sorted({node for src, dst, _ in edges for node in (src, dst)})
            scores = pagerank_scores(
                edges,
                nodes,
                nodes=nodes,
                restart=1.0 - cfg.pagerank_damping,
                tolerance=cfg.pagerank_tolerance,
                max_iterations=cfg.pagerank_iterations,
            )
            db.executemany(
                "INSERT INTO node_pagerank(node_entity_id,graph_kind,score,build_id) "
                "VALUES(?,?,?,?)",
                ((node, kind, score, build_id) for node, score in sorted(scores.items())),
            )
            result[kind] = len(scores)
    return result


def materialize_task_pagerank(
    database: Path,
    *,
    build_id: str,
    top_k: int = 8,
    radius: int = 2,
    maximum_task_nodes: int = 256,
    restart: float = 0.15,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> dict[str, int]:
    """Materialize bounded PPR: similar-work projection and real credit graphs.

    Every seed is expanded only inside its task graph and bounded ego network;
    runtime search therefore remains a table lookup rather than an online graph walk.
    voice_cast admits person/character/subject seeds; staff_filmography admits
    person/subject seeds. Results contain mixed entity types and a sourced path,
    not an assertion of direct collaboration for multi-hop neighbors.
    """

    from mikirin.graph.algorithms import GraphEdge, pagerank_scores
    from mikirin.graph.tasks import (
        HETEROGENEOUS_TASKS, TASK_GRAPH_VERSION, credit_path, load_credit_graph,
    )

    if (any(isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (top_k, radius, maximum_task_nodes))
            or maximum_task_nodes < top_k + 1):
        raise ValueError("invalid task PageRank limits")
    if not 0 < restart < 1:
        raise ValueError("restart must be in (0,1)")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations < 1 or not math.isfinite(tolerance) or tolerance < 0):
        raise ValueError("invalid task PageRank convergence parameters")
    tasks: dict[str, tuple[str, ...]] = {
        "similar_work": ("tag_similarity", "same_season"),
        **HETEROGENEOUS_TASKS,
    }
    counts: dict[str, int] = {}
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        ensure_p1_schema(db)
        if db.execute("SELECT 1 FROM build WHERE id=?", (build_id,)).fetchone() is None:
            raise KeyError(f"unknown build_id: {build_id}")
        db.execute("DELETE FROM task_pagerank WHERE build_id=?", (build_id,))
        for graph_kind, edge_kinds in tasks.items():
            credit_graph = (load_credit_graph(db, graph_kind)
                            if graph_kind in HETEROGENEOUS_TASKS else None)
            marks = ",".join("?" for _ in edge_kinds)
            rows = credit_graph.edges if credit_graph is not None else [
                (str(src), str(dst), float(weight), str(kind))
                for src, dst, weight, kind in db.execute(
                    f"SELECT src_entity_id,dst_entity_id,weight,kind FROM derived_edge "
                    f"WHERE build_id=? AND kind IN ({marks}) "
                    "ORDER BY src_entity_id,dst_entity_id,kind",
                    (build_id, *edge_kinds),
                )
            ]
            adjacency: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
            for src, dst, weight, _ in rows:
                adjacency[src][dst] += weight
                adjacency[dst][src] += weight
            inserted = 0
            batch: list[tuple[Any, ...]] = []
            for seed in sorted(adjacency):
                task_nodes = {seed}
                frontier = {seed}
                parents: dict[str, str] = {}
                for _ in range(radius):
                    candidates: dict[str, float] = defaultdict(float)
                    candidate_parents: dict[str, str] = {}
                    for node in sorted(frontier):
                        for neighbor, weight in adjacency[node].items():
                            if neighbor not in task_nodes:
                                candidates[neighbor] += weight
                                candidate_parents.setdefault(neighbor, node)
                    room = maximum_task_nodes - len(task_nodes)
                    if room <= 0 or not candidates:
                        break
                    selected = {
                        node
                        for node, _ in sorted(
                            candidates.items(), key=lambda item: (-item[1], item[0])
                        )[:room]
                    }
                    task_nodes.update(selected)
                    parents.update({node: candidate_parents[node] for node in selected})
                    frontier = selected
                task_edges: list[GraphEdge] = []
                for src in sorted(task_nodes):
                    for dst, weight in sorted(adjacency[src].items()):
                        if src < dst and dst in task_nodes:
                            task_edges.append(GraphEdge(src, dst, weight, graph_kind))
                score_map = pagerank_scores(
                    task_edges,
                    {seed: 1.0},
                    task_nodes=task_nodes,
                    restart=restart,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                )
                ranked = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))[
                    : top_k + 1
                ]
                rank = 0
                for node, score in ranked:
                    if node == seed:
                        continue
                    rank += 1
                    if rank > top_k:
                        break
                    evidence = {
                        "edge_kinds": edge_kinds, "radius": radius,
                        "maximum_task_nodes": maximum_task_nodes, "restart": restart,
                        "max_iterations": max_iterations, "tolerance": tolerance,
                    }
                    if credit_graph is not None:
                        evidence.update(
                            graph_version=TASK_GRAPH_VERSION, top_k=top_k,
                            seed_entity_type=credit_graph.node_types[seed],
                            node_entity_type=credit_graph.node_types[node],
                            task_node_count=len(task_nodes),
                            path=credit_path(credit_graph, parents, seed, node),
                            attribution="显式演职员关系；多跳路径不代表种子与结果有直接合作关系。",
                        )
                    batch.append(
                        (
                            seed,
                            node,
                            graph_kind,
                            rank,
                            score,
                            json.dumps(
                                evidence,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                            build_id,
                        )
                    )
                if len(batch) >= 20_000:
                    db.executemany("INSERT INTO task_pagerank VALUES(?,?,?,?,?,?,?)", batch)
                    inserted += len(batch)
                    batch.clear()
            if batch:
                db.executemany("INSERT INTO task_pagerank VALUES(?,?,?,?,?,?,?)", batch)
                inserted += len(batch)
            counts[graph_kind] = inserted
    return counts


def deterministic_edge_digest(database: Path, *, build_id: str) -> str:
    digest = hashlib.sha256()
    uri = database.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        for row in db.execute(
            "SELECT src_entity_id,dst_entity_id,kind,printf('%.12g',weight),evidence_json "
            "FROM derived_edge WHERE build_id=? ORDER BY kind,src_entity_id,dst_entity_id",
            (build_id,),
        ):
            digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def build_manifest(
    *, paths: BuildPaths, config: P1BuildConfig, stages: dict[str, Any]
) -> dict[str, Any]:
    return {
        "processor_version": P1_PROCESSOR_VERSION,
        "schema_version": CONTENT_SCHEMA_VERSION,
        "build_id": paths.build_id,
        "version": paths.version,
        "source_database": paths.source.name,
        "target_database": paths.target.name,
        "config": asdict(config),
        "stages": stages,
        "generated_at": utc_now(),
    }

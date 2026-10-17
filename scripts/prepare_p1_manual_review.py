"""Generate blank, deterministic P1 manual-review packages.

This is intentionally a data-only tool.  It samples rows from a side database,
but never turns a score, model decision, relation, or search result into a
human label.  The ``sampling_stratum`` column in the A4 calibration file is
only the deterministic high/low-score sampling rule; ``label`` remains empty.

Usage::

    python scripts/prepare_p1_manual_review.py SIDE_DB BUILD_ID SLANG_RUN_ID OUT_DIR

The output files are written with sibling temporary files and ``os.replace``.
All data sufficiency checks happen before the first output is written.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sqlite3
import sys
from collections.abc import Iterable, Mapping
from contextlib import closing
from pathlib import Path
from typing import Any

# Allow direct invocation from any directory without an editable install.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if SOURCE_ROOT.is_dir() and str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mikirin.atomic import atomic_write_json, atomic_write_text  # noqa: E402
from mikirin.graph.search import build_ab_package  # noqa: E402


DEFAULT_SEED = 20260802
CALIBRATION_EACH = 100
GOLDEN_QUERY_COUNT = 50
IMPLICIT_EDGE_COUNT = 100
AB_QUERY_COUNT = 30


class PreparationError(RuntimeError):
    """Raised when the side database cannot supply a complete review package."""


def _read_only_connection(database: Path) -> sqlite3.Connection:
    if not database.is_file():
        raise PreparationError(f"side DB does not exist: {database}")
    try:
        return sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise PreparationError(f"cannot open side DB read-only: {database}: {error}") from error


def _require_tables(db: sqlite3.Connection, tables: Iterable[str]) -> None:
    available = {
        str(row[0])
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    missing = sorted(set(tables) - available)
    if missing:
        raise PreparationError("side DB is missing required tables: " + ", ".join(missing))


def _check_identity(db: sqlite3.Connection, build_id: str, slang_run_id: str) -> None:
    if db.execute("SELECT 1 FROM build WHERE id=?", (build_id,)).fetchone() is None:
        raise PreparationError(f"unknown build id: {build_id}")
    run = db.execute(
        "SELECT status FROM slang_run WHERE id=? AND build_id=?",
        (slang_run_id, build_id),
    ).fetchone()
    if run is None:
        raise PreparationError(
            f"slang run {slang_run_id!r} is absent or belongs to another build ({build_id!r})"
        )
    if str(run[0]).casefold() not in {"completed", "complete", "done"}:
        raise PreparationError(f"slang run is not complete: {slang_run_id} (status={run[0]!r})")


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _csv_text(fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=fieldnames,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({name: "" if row.get(name) is None else row[name] for name in fieldnames})
    return buffer.getvalue()


def _candidate_rows(db: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    db.row_factory = sqlite3.Row
    rows = db.execute(
        """
        SELECT id,candidate_text,normalized_candidate,bucket,occurrence_count,
               document_count,author_count,feature_json,score,threshold,status
        FROM slang_candidate
        WHERE run_id=?
        ORDER BY score DESC, normalized_candidate ASC, id ASC
        """,
        (run_id,),
    ).fetchall()
    return list(rows)


def _make_calibration_rows(db: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    candidates = _candidate_rows(db, run_id)
    required = CALIBRATION_EACH * 2
    if len(candidates) < required:
        raise PreparationError(
            f"A4 calibration needs {required} slang candidates for two {CALIBRATION_EACH}-row "
            f"sampling strata; found {len(candidates)} for run {run_id}"
        )

    selected = [
        ("high_score", row) for row in candidates[:CALIBRATION_EACH]
    ] + [
        ("low_score", row) for row in candidates[-CALIBRATION_EACH:]
    ]
    rows: list[dict[str, Any]] = []
    for stratum, row in selected:
        evidence = [
            str(item[0])
            for item in db.execute(
                "SELECT context_text FROM slang_evidence "
                "WHERE candidate_id=? ORDER BY id LIMIT 3",
                (row["id"],),
            ).fetchall()
        ]
        rows.append(
            {
                "candidate_id": row["id"],
                "candidate_text": row["candidate_text"],
                "normalized_candidate": row["normalized_candidate"],
                "bucket": row["bucket"],
                "occurrence_count": row["occurrence_count"],
                "document_count": row["document_count"],
                "author_count": row["author_count"],
                "score": row["score"],
                "threshold": row["threshold"],
                "status": row["status"],
                "feature_json": row["feature_json"],
                "evidence_contexts": _json_cell(evidence),
                "sampling_stratum": stratum,
                "label": "",
                "review_notes": "",
            }
        )
    return rows


def _make_golden_rows(db: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    db.row_factory = sqlite3.Row
    rows = db.execute(
        """
        SELECT id,candidate_text,normalized_candidate,bucket,score
        FROM slang_candidate
        WHERE run_id=? AND TRIM(candidate_text)<>''
        ORDER BY score DESC, normalized_candidate ASC, id ASC
        """,
        (run_id,),
    ).fetchall()
    unique: list[sqlite3.Row] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row["normalized_candidate"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
        if len(unique) == GOLDEN_QUERY_COUNT:
            break
    if len(unique) < GOLDEN_QUERY_COUNT:
        raise PreparationError(
            f"A4 slang golden set needs at least {GOLDEN_QUERY_COUNT} distinct non-empty "
            f"queries; found {len(unique)} for run {run_id}"
        )
    return [
        {
            "query_id": f"slang-golden-{index:03d}",
            "query": row["candidate_text"],
            "candidate_id": row["id"],
            "sampling_score": row["score"],
            "expected": "",
            "review_notes": "",
        }
        for index, row in enumerate(unique, 1)
    ]


def _make_implicit_rows(
    db: sqlite3.Connection, build_id: str, run_id: str
) -> list[dict[str, Any]]:
    db.row_factory = sqlite3.Row
    rows = db.execute(
        """
        SELECT ie.src_entity_id,ie.dst_entity_id,
               COALESCE(NULLIF(src.canonical_name,''),ie.src_entity_id) AS src_name,
               COALESCE(NULLIF(dst.canonical_name,''),ie.dst_entity_id) AS dst_name,
               ie.relation,ie.sentiment,ie.weight,ie.evidence_count,
               ie.confidence,ie.provenance_json
        FROM implicit_edge ie
        JOIN entity src ON src.id=ie.src_entity_id
        JOIN entity dst ON dst.id=ie.dst_entity_id
        WHERE ie.build_id=?
        ORDER BY ie.src_entity_id,ie.dst_entity_id,ie.relation,ie.sentiment
        """,
        (build_id,),
    ).fetchall()
    if len(rows) < IMPLICIT_EDGE_COUNT:
        raise PreparationError(
            f"A5 implicit-edge review needs at least {IMPLICIT_EDGE_COUNT} materialized edges "
            f"for build {build_id}; found {len(rows)}"
        )
    selected = rows[:IMPLICIT_EDGE_COUNT]
    result: list[dict[str, Any]] = []
    for index, row in enumerate(selected, 1):
        evidence = [
            {
                "evidence_id": item[0],
                "document_id": item[1],
                "claim_quote": item[2],
                "author_hash": item[3],
            }
            for item in db.execute(
                """
                SELECT id,document_id,claim_quote,author_hash
                FROM implicit_edge_evidence
                WHERE run_id=? AND src_entity_id=? AND dst_entity_id=?
                  AND relation=? AND sentiment=?
                ORDER BY id LIMIT 3
                """,
                (
                    run_id,
                    row["src_entity_id"],
                    row["dst_entity_id"],
                    row["relation"],
                    row["sentiment"],
                ),
            ).fetchall()
        ]
        result.append(
            {
                "review_id": f"implicit-edge-{index:03d}",
                "src_entity_id": row["src_entity_id"],
                "dst_entity_id": row["dst_entity_id"],
                "src_name": row["src_name"],
                "dst_name": row["dst_name"],
                "relation": row["relation"],
                "sentiment": row["sentiment"],
                "weight": row["weight"],
                "evidence_count": row["evidence_count"],
                "confidence": row["confidence"],
                "provenance_json": row["provenance_json"],
                "evidence": _json_cell(evidence),
                "label": "",
                "review_notes": "",
            }
        )
    return result


def _make_ab_package(
    db: sqlite3.Connection, build_id: str, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    db.row_factory = sqlite3.Row
    rows = db.execute(
        """
        SELECT s.entity_id,COALESCE(NULLIF(s.name_cn,''),s.name) AS query
        FROM task_pagerank tp
        JOIN subjects s ON s.entity_id=tp.seed_entity_id
        WHERE tp.build_id=? AND tp.graph_kind='similar_work'
          AND TRIM(COALESCE(NULLIF(s.name_cn,''),s.name))<>''
        GROUP BY s.entity_id,query
        ORDER BY s.entity_id
        """,
        (build_id,),
    ).fetchall()
    if len(rows) < AB_QUERY_COUNT:
        raise PreparationError(
            f"A5 anonymous A/B package needs at least {AB_QUERY_COUNT} query seeds with "
            f"similar_work PageRank for build {build_id}; found {len(rows)}"
        )
    pool = [
        (
            str(row["query"]),
            {"seed_entity_ids": [str(row["entity_id"])], "task": "similar_work"},
        )
        for row in rows
    ]
    randomizer = random.Random(seed)
    randomizer.shuffle(pool)
    package = build_ab_package(
        [{"query": query, **details} for query, details in pool[:AB_QUERY_COUNT]],
        arm_a="simple",
        arm_b="fusion",
    ).to_dict()
    csv_rows = [
        {
            "query_id": item["query_id"],
            "query": item["query"],
            "seed_entity_ids": _json_cell(item["seed_entity_ids"]),
            "task": item["task"] or "",
            "arm_a": package["arms"]["a"],
            "arm_b": package["arms"]["b"],
            "human_preference": "",
            "review_notes": "",
        }
        for item in package["queries"]
    ]
    package["generation"] = {
        "seed": seed,
        "query_count": AB_QUERY_COUNT,
    }
    return package, csv_rows


def prepare_p1_manual_review(
    side_db: Path | str,
    build_id: str,
    slang_run_id: str,
    output_dir: Path | str,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    """Generate all P1 review files and return their paths.

    The function performs every source-data and row-count check before writing.
    Existing destination files may be replaced atomically, but no database or
    source table is modified.
    """

    database = Path(side_db)
    destination = Path(output_dir)
    try:
        with closing(_read_only_connection(database)) as db:
            _require_tables(
                db,
                (
                    "build",
                    "slang_run",
                    "slang_candidate",
                    "slang_evidence",
                    "implicit_edge",
                    "implicit_edge_evidence",
                    "entity",
                    "subjects",
                    "task_pagerank",
                ),
            )
            _check_identity(db, build_id, slang_run_id)
            calibration = _make_calibration_rows(db, slang_run_id)
            golden = _make_golden_rows(db, slang_run_id)
            implicit = _make_implicit_rows(db, build_id, slang_run_id)
            ab_package, ab_csv = _make_ab_package(db, build_id, seed)
    except PreparationError:
        raise
    except sqlite3.Error as error:
        raise PreparationError(f"cannot read P1 side DB: {error}") from error

    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "a4_calibration": destination / "a4_calibration.csv",
        "a4_golden_queries": destination / "a4_golden_queries.csv",
        "a5_implicit_edge_review": destination / "a5_implicit_edge_review.csv",
        "a5_query_ab_csv": destination / "a5_query_ab.csv",
        "a5_query_ab_json": destination / "a5_query_ab.json",
    }
    atomic_write_text(
        outputs["a4_calibration"],
        _csv_text(
            [
                "candidate_id",
                "candidate_text",
                "normalized_candidate",
                "bucket",
                "occurrence_count",
                "document_count",
                "author_count",
                "score",
                "threshold",
                "status",
                "feature_json",
                "evidence_contexts",
                "sampling_stratum",
                "label",
                "review_notes",
            ],
            calibration,
        ),
    )
    atomic_write_text(
        outputs["a4_golden_queries"],
        _csv_text(
            ["query_id", "query", "candidate_id", "sampling_score", "expected", "review_notes"],
            golden,
        ),
    )
    atomic_write_text(
        outputs["a5_implicit_edge_review"],
        _csv_text(
            [
                "review_id",
                "src_entity_id",
                "dst_entity_id",
                "src_name",
                "dst_name",
                "relation",
                "sentiment",
                "weight",
                "evidence_count",
                "confidence",
                "provenance_json",
                "evidence",
                "label",
                "review_notes",
            ],
            implicit,
        ),
    )
    atomic_write_text(
        outputs["a5_query_ab_csv"],
        _csv_text(
            [
                "query_id",
                "query",
                "seed_entity_ids",
                "task",
                "arm_a",
                "arm_b",
                "human_preference",
                "review_notes",
            ],
            ab_csv,
        ),
    )
    atomic_write_json(outputs["a5_query_ab_json"], ab_package)
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("side_db", type=Path, help="P1 side SQLite database")
    parser.add_argument("build_id")
    parser.add_argument("slang_run_id")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        outputs = prepare_p1_manual_review(
            args.side_db,
            args.build_id,
            args.slang_run_id,
            args.output_dir,
            seed=args.seed,
        )
    except PreparationError as error:
        _parser().error(str(error))
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

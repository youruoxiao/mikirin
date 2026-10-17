"""Offline P1 evaluation. Blank labels are unknown, never inferred from scores.

All database access is read-only. Candidate retrieval is supplied by the caller;
this module never calls an embedding/model service or enables experimental search.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
from collections.abc import Callable, Iterable, Mapping
from contextlib import closing
from importlib.resources import files
from pathlib import Path
from typing import Any

from mikirin.atomic import atomic_write_json
from mikirin.graph.search import OfflineGraphSearch


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def _unique(rows: list[dict], key: str) -> None:
    ids = [row.get(key) for row in rows]
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError(f"each row requires a nonempty {key}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate {key}")


def _label(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if value is True or value in ("positive", "correct", "1"):
        return True
    if value is False or value in ("negative", "incorrect", "0"):
        return False
    raise ValueError("human label must be positive/negative, correct/incorrect, 1/0 or blank")


def _score(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("score must be a finite number in [0,1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("score must be a finite number in [0,1]")
    return result


def load_graph_queries(path: Path | str | None = None) -> list[dict[str, Any]]:
    source = Path(path) if path else files("mikirin").joinpath("resources/benchmarks/graph-30.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload["queries"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or len(rows) != 30:
        raise ValueError("graph comparison requires exactly 30 queries")
    _unique(rows, "query_id")
    if any(not isinstance(row.get("query"), str) or not row["query"].strip() for row in rows):
        raise ValueError("query text is required")
    return rows


def _blind_results(searcher: OfflineGraphSearch, results: list[dict], spec: dict) -> list[dict]:
    """Use the same display schema for every arm; omit ranking internals."""
    names = {}
    ids = [hit["entity_id"] for hit in results]
    if ids:
        with searcher._read_connection() as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='entity' AND type='table'").fetchone():
                names = dict(db.execute("SELECT id,canonical_name FROM entity WHERE id IN (" +
                                        ",".join("?" for _ in ids) + ")", ids))
    supplied: dict[str, dict] = {}
    for key in ("vector_candidates", "lexical_candidates"):
        for item in spec.get(key, ()):
            if isinstance(item, Mapping):
                supplied.setdefault(item.get("entity_id"), {}).update(item)
    displayed = []
    for hit in results:
        original = supplied.get(hit["entity_id"], {})
        evidence = []
        sources = [original, *(original.get("citations") or [])]
        for provenance in hit.get("provenance", []):
            raw = provenance.get("evidence", {})
            if isinstance(raw, Mapping):
                sources.append(raw)
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            item = {key: source[key] for key in ("document_id", "url", "source_url", "quote",
                                                 "claim_quote", "author_hash", "attribution")
                    if source.get(key) is not None}
            if item and item not in evidence:
                evidence.append(item)
        displayed.append({"entity_id": hit["entity_id"],
                          "canonical_name": names.get(hit["entity_id"]) or original.get("canonical_name") or
                                            original.get("name") or f"名称未提供（{hit['entity_id']}）",
                          "evidence": evidence,
                          "evidence_note": "来源由检索候选提供，需回原文核对" if evidence else
                                           "该结果未提供可逐字定位的引文；不能据此当作社区共识。"})
    return displayed


def run_graph_comparison(
    database: Path | str | sqlite3.Connection,
    queries: Iterable[Mapping[str, Any]], *, build_id: str,
    output_dir: Path | str | None = None, seed: int = 0,
    index_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Run simple / graph / fusion on frozen inputs, with separate blind/private files.

    Query fields: query_id, query, task (similar_work, voice_cast or
    staff_filmography), seed_entity_ids, lexical_candidates, vector_candidates.
    Gold expected_entity_ids may be null (unlabeled), [] (expect abstention),
    or a nonempty list supplied by a human. Missing retrieval inputs are reported.
    """
    specs = [dict(row) for row in queries]
    if len(specs) != 30:
        raise ValueError("graph comparison requires exactly 30 queries")
    _unique(specs, "query_id")
    _unique(specs, "query")
    if not build_id:
        raise ValueError("build_id is required")
    if output_dir is not None:
        destination = Path(output_dir)
        if any((destination / f"graph-{name}.json").exists() for name in ("private", "blind", "metrics")):
            raise FileExistsError("evaluation outputs already exist; use a new output directory")
    searcher = OfflineGraphSearch(database)
    private_rows, blind_rows = [], []
    totals = {mode: {"queries": 0, "nonempty": 0, "labeled": 0, "top1_hits": 0,
                     "recall_at_8_sum": 0.0} for mode in ("simple", "graph", "fusion")}
    for spec in specs:
        if not str(spec.get("query", "")).strip():
            raise ValueError("query text is required")
        task = spec.get("task", "similar_work")
        if task not in {"similar_work", "voice_cast", "staff_filmography"}:
            raise ValueError(f"unknown task subgraph: {task}")
        gold = spec.get("expected_entity_ids")
        if gold is not None and (not isinstance(gold, list) or
                                 any(not isinstance(item, str) or not item for item in gold)):
            raise ValueError("expected_entity_ids must be null or a list of entity IDs")
        results = {}
        for mode in totals:
            hits = searcher.search(
                spec.get("seed_entity_ids", ()), mode=mode, task=task, build_id=build_id,
                lexical_candidates=spec.get("lexical_candidates", ()),
                vector_candidates=spec.get("vector_candidates", ()) if mode == "fusion" else (),
            )
            results[mode] = [hit.to_dict() for hit in hits]
            total = totals[mode]
            total["queries"] += 1
            total["nonempty"] += bool(hits)
            if gold is not None:
                total["labeled"] += 1
                total["top1_hits"] += bool(hits and hits[0].entity_id in gold) if gold else not hits
                total["recall_at_8_sum"] += (len({hit.entity_id for hit in hits} & set(gold)) /
                                             len(set(gold))) if gold else float(not hits)
        order = sorted(results)
        random.Random(_digest([seed, spec["query_id"]])).shuffle(order)
        mapping = dict(zip(("A", "B", "C"), order))
        blind_rows.append({
            "query_id": spec["query_id"], "query": spec["query"],
            "arms": {arm: {"results": _blind_results(searcher, results[mode], spec),
                            "relevance": None, "surprise": None, "review_notes": ""}
                     for arm, mode in mapping.items()},
        })
        private_rows.append({"query_id": spec["query_id"], "arm_mapping": mapping,
                             "results": results, "inputs": spec,
                             "missing_inputs": [key for key in ("seed_entity_ids", "lexical_candidates",
                                                                "vector_candidates") if key not in spec]})
    metrics = {mode: {"query_count": values["queries"], "nonempty_count": values["nonempty"],
                      "labeled_count": values["labeled"],
                      "top1_accuracy": values["top1_hits"] / values["labeled"] if values["labeled"] else None,
                      "recall_at_8": values["recall_at_8_sum"] / values["labeled"] if values["labeled"] else None}
               for mode, values in totals.items()}
    identity = {"version": "p1-evaluation-v1", "build_id": build_id,
                "index_fingerprint": index_fingerprint, "input_digest": _digest(specs),
                "seed": seed, "restart": 0.15, "top_k": 8,
                "fusion_weights": [0.45, 0.25, 0.30]}
    result = {"private": {**identity, "queries": private_rows},
              "blind": {"experiment_id": _digest(identity), "queries": blind_rows},
              "metrics": {**identity, "modes": metrics, "human_acceptance": None,
                          "default_change_authorized": False,
                          "vector_inputs_complete": all("vector_candidates" in row for row in specs)}}
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name, payload in result.items():
            if (destination / f"graph-{name}.json").exists():
                raise FileExistsError("evaluation output appeared while running; refusing to overwrite")
            atomic_write_json(destination / f"graph-{name}.json", payload)
    return result


def evaluate_graph_ratings(blind: Mapping[str, Any], private: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize explicit human 1..5 relevance/surprise ratings; no auto rollout."""
    if blind.get("experiment_id") != _digest({key: value for key, value in private.items() if key != "queries"}):
        raise ValueError("blind/private experiment identity mismatch")
    rows = list(blind["queries"])
    _unique(rows, "query_id")
    mapping = {row["query_id"]: row["arm_mapping"] for row in private["queries"]}
    scores = {mode: {dimension: [] for dimension in ("relevance", "surprise")}
              for mode in ("simple", "graph", "fusion")}
    if set(mapping) != {row["query_id"] for row in rows}:
        raise ValueError("ratings query IDs differ from experiment")
    complete = True
    for row in rows:
        if set(row["arms"]) != set(mapping[row["query_id"]]):
            raise ValueError("ratings arms differ from experiment")
        for arm, values in row["arms"].items():
            for dimension in ("relevance", "surprise"):
                value = values.get(dimension)
                if value is None or value == "":
                    complete = False
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 5:
                    raise ValueError("human ratings must be in [1,5] or blank")
                scores[mapping[row["query_id"]][arm]][dimension].append(value)
    return {"status": "rated" if complete else "awaiting_human_labels",
            "modes": {mode: {key: {"count": len(values), "mean": sum(values) / len(values) if values else None}
                              for key, values in dimensions.items()} for mode, dimensions in scores.items()},
            "default_change_authorized": False}


def calibrate_slang_thresholds(rows: Iterable[Mapping[str, Any]], *, threshold_version: str) -> dict[str, Any]:
    """Suggest conservative score thresholds from >=100 positive +100 negative labels.

    This is an in-sample calibration report, not held-out quality acceptance.
    The version and input digest must accompany any applied configuration.
    """
    data = [dict(row) for row in rows]
    _unique(data, "candidate_id")
    if not threshold_version.strip():
        raise ValueError("threshold_version is required")
    labeled = [(_score(row["score"]), _label(row.get("label")), row.get("bucket")) for row in data]
    positives = sum(label is True for _, label, _ in labeled)
    negatives = sum(label is False for _, label, _ in labeled)
    report = {"threshold_version": threshold_version, "input_digest": _digest(data),
              "positive_count": positives, "negative_count": negatives,
              "unlabeled_count": len(data) - positives - negatives,
              "status": "awaiting_human_labels", "thresholds": None,
              "held_out_acceptance": None}
    if positives < 100 or negatives < 100:
        return report

    def choose(values: list[tuple]) -> float | None:
        positive = [score for score, label, _ in values if label is True]
        negative = [score for score, label, _ in values if label is False]
        if not positive or not negative:
            return None
        return next((threshold for threshold in sorted(set(positive))
                     if all(score < threshold for score in negative)), None)

    general = choose([row for row in labeled if row[2] != "single_char"])
    single = choose([row for row in labeled if row[2] == "single_char"])
    if general is not None and single is not None:
        single = max(general, single)
    report.update(status="calibrated" if general is not None else "no_safe_threshold", thresholds={"candidate_score_threshold": general,
                                                  "single_char_score_threshold": single},
                  note="Null thresholds require more labels or indicate no zero-false-positive cutoff; do not auto-apply.")
    return report


def evaluate_slang_golden(rows: Iterable[Mapping[str, Any]], *, search: Callable[[str], Iterable[Any]]) -> dict[str, Any]:
    data = [dict(row) for row in rows]
    _unique(data, "query_id")
    _unique(data, "query")
    results, hit_count, wrong_count, labeled_count = [], 0, 0, 0
    single_correct = single_returned = single_labeled = 0
    for row in data:
        expected = row.get("expected_entity_ids", row.get("expected"))
        if isinstance(expected, str) and expected.strip():
            expected = json.loads(expected) if expected.lstrip().startswith("[") else [expected]
        elif expected == "":
            expected = None
        if expected is not None and (not isinstance(expected, list) or any(not isinstance(x, str) or not x for x in expected)):
            raise ValueError("expected must be blank or entity IDs (JSON list supports abstention [])")
        hits = list(search(str(row["query"])))
        ids = [str(hit if isinstance(hit, str) else hit.get("entity_id", hit.get("id", "")))
               if isinstance(hit, (str, Mapping)) else str(hit.entity_id) for hit in hits]
        if any(not value for value in ids):
            raise ValueError("search results require entity_id")
        correct = None if expected is None else (ids[0] in expected if ids else not expected)
        if expected is not None:
            labeled_count += 1
            hit_count += bool(correct)
            wrong_count += len(set(ids) - set(expected))
            if len(str(row["query"]).strip()) == 1:
                single_labeled += 1
                single_returned += bool(ids)
                single_correct += bool(ids and correct)
        results.append({"query_id": row["query_id"], "query": row["query"], "results": ids, "top1_correct": correct})
    accuracy = hit_count / labeled_count if labeled_count else None
    precision = single_correct / single_returned if single_returned else None
    ready = labeled_count >= 50 and labeled_count == len(data)
    return {"status": "evaluated" if ready else "awaiting_human_labels", "input_digest": _digest(data),
            "labeled_count": labeled_count, "top1_accuracy": accuracy, "wrong_hits": wrong_count,
            "single_char_labeled_count": single_labeled, "single_char_precision": precision,
            "acceptance_passed": bool(ready and accuracy >= 0.90 and wrong_count == 0 and
                                      precision is not None and precision >= 0.95), "queries": results}


def prepare_implicit_review(database: Path | str, *, build_id: str, sample_size: int = 100, seed: int = 0) -> list[dict[str, Any]]:
    """Deterministically sample edges and verify every attributed source quote."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(row) for row in db.execute("SELECT * FROM implicit_edge WHERE build_id=? ORDER BY src_entity_id,dst_entity_id,relation,sentiment", (build_id,))]
        random.Random(seed).shuffle(rows)
        result = []
        for row in rows[:sample_size]:
            provenance = json.loads(row["provenance_json"])
            evidence = []
            for item in db.execute(
                "SELECT e.id,e.document_id,e.claim_quote,e.char_start,e.char_end,e.author_hash,d.content "
                "FROM implicit_edge_evidence e LEFT JOIN source_document d ON d.id=e.document_id "
                "WHERE e.run_id=? AND e.src_entity_id=? AND e.dst_entity_id=? AND e.relation=? AND e.sentiment=? ORDER BY e.id",
                (provenance.get("run_id"), row["src_entity_id"], row["dst_entity_id"], row["relation"], row["sentiment"]),
            ):
                item = dict(item)
                text, quote = item.pop("content"), item["claim_quote"]
                start, end = item["char_start"], item["char_end"]
                item["quote_verified"] = bool(text is not None and quote and 0 <= start < end <= len(text) and text[start:end] == quote)
                item["source_excerpt"] = text[max(0, start - 120):end + 120] if text else None
                evidence.append(item)
            row.update(review_id=_digest([build_id, row["src_entity_id"], row["dst_entity_id"], row["relation"], row["sentiment"]]),
                       evidence=evidence, quote_verified=bool(evidence) and all(item["quote_verified"] for item in evidence),
                       label=None, review_notes="")
            result.append(row)
    return result


def evaluate_implicit_review(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    data = [dict(row) for row in rows]
    _unique(data, "review_id")
    labels = [_label(row.get("label")) for row in data]
    labeled = sum(value is not None for value in labels)
    precision = sum(value is True for value in labels) / labeled if labeled else None
    verified = sum(row.get("quote_verified") is True for row in data)
    ready = len(data) >= 100 and labeled == len(data)
    return {"status": "evaluated" if ready else "awaiting_human_labels", "sample_count": len(data),
            "labeled_count": labeled, "human_precision": precision, "quote_verified_count": verified,
            "quote_verification_rate": verified / len(data) if data else None,
            "acceptance_passed": bool(ready and precision >= 0.85 and verified == len(data)),
            "input_digest": _digest(data)}


def prepare_monthly_alias_review(database: Path | str, *, month: str, sample_size: int = 50, seed: int = 0) -> list[dict[str, Any]]:
    """Return a reproducible blank monthly audit; fewer than 50 is not acceptance."""
    import datetime
    datetime.datetime.strptime(month, "%Y-%m")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(row) for row in db.execute("SELECT * FROM entity_alias WHERE source_kind='mined' AND status IN ('confirmed','provisional') ORDER BY id")]
    random.Random(_digest([month, seed])).shuffle(rows)
    return [{**row, "review_id": _digest([month, row["id"]]), "month": month,
             "label": None, "review_notes": ""} for row in rows[:sample_size]]

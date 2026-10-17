"""Versioned retrieval experiments. Missing annotations stay null, never zero.

CLI: python -m mikirin.retrieval_evaluation --database content.db
Use --semantic --root PROJECT only to explicitly allow query embeddings.
Gold schema: scope={subject_entity_ids, episode_entity_ids} (acceptable IDs),
evidence_document_ids, dissenting_document_ids, citation_labels={document_id: bool}.
Annotations must come from a human; this module does not generate them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from mikirin.hybrid_retrieval import HybridRetriever, HybridSemanticSearch, RETRIEVAL_MODES

METRICS = ("scope_accuracy", "evidence_recall@8", "citation_precision",
           "contradiction_coverage", "author_diversity")


def load_benchmark(path: Path | str | None = None) -> dict[str, Any]:
    """Read the default installed-package 60-question template or an explicit file."""
    text = (Path(path).read_text(encoding="utf-8") if path is not None else
            resources.files("mikirin").joinpath("resources", "benchmarks", "retrieval-60.json").read_text(encoding="utf-8"))
    benchmark = json.loads(text)
    _validate(benchmark)
    return benchmark


def _validate(benchmark: dict[str, Any]) -> None:
    if not isinstance(benchmark, dict) or not benchmark.get("version"):
        raise ValueError("benchmark requires a nonempty version")
    questions = benchmark.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("benchmark requires questions")
    ids = set()
    for question in questions:
        if not isinstance(question, dict) or not isinstance(question.get("id"), str) or not question["id"] or question["id"] in ids:
            raise ValueError("question IDs must be nonempty and unique")
        if not isinstance(question.get("query"), str) or not question["query"].strip():
            raise ValueError("question requires query text")
        if question.get("gold") is not None and not isinstance(question["gold"], dict):
            raise ValueError("gold must be an object or null")
        ids.add(question["id"])


def _ids(value: Any) -> set[str] | None:
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
        return None
    return set(value)


def _metrics(question: dict, result: dict, database: Path) -> tuple[dict, dict]:
    metrics = dict.fromkeys(METRICS)
    reasons = {key: "human_gold_missing" for key in METRICS}
    gold = question.get("gold") or {}
    # Merely filling IDs does not make them human gold; require provenance.
    annotation = question.get("annotation") or {}
    approved_gold = (annotation.get("source") == "human" and bool(annotation.get("annotator")))
    if not approved_gold:
        gold = {}
    documents = result.get("documents", [])
    retrieved = list(dict.fromkeys(d["document_id"] for d in documents))[:8]
    expected = _ids(gold.get("evidence_document_ids"))
    if expected is not None:
        metrics["evidence_recall@8"] = len(expected.intersection(retrieved)) / len(expected)
    scope = gold.get("scope") or {}
    scope_checks = []
    for field, results, id_key in (("subject_entity_ids", result.get("subjects", []), "subject_entity_id"),
                                    ("episode_entity_ids", result.get("episodes", []), "episode_entity_id")):
        acceptable = _ids(scope.get(field))
        if acceptable is not None:
            scope_checks.append(bool(results and results[0][id_key] in acceptable))
    if scope_checks:
        metrics["scope_accuracy"] = float(all(scope_checks))
    dissent = _ids(gold.get("dissenting_document_ids"))
    if dissent is not None:
        metrics["contradiction_coverage"] = len(dissent.intersection(retrieved)) / len(dissent)
    labels = gold.get("citation_labels")
    if isinstance(labels, dict) and retrieved and all(isinstance(labels.get(key), bool) for key in retrieved):
        metrics["citation_precision"] = sum(labels[key] for key in retrieved) / len(retrieved)
    elif isinstance(labels, dict) and retrieved:
        reasons["citation_precision"] = "incomplete_human_citation_labels"
    known = [d["author_hash"] for d in documents[:8] if d.get("author_hash")]
    metrics["author_diversity"] = len(set(known)) / len(known) if known else None
    reasons["author_diversity"] = "no_known_authors" if not known else None
    # Independently verify raw excerpts. This is source fidelity, NOT human
    # judgment that a quotation supports an answer or is relevant to the query.
    checks = []
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for item in documents[:8]:
            row = db.execute("SELECT content, url FROM source_document WHERE id=?", (item["document_id"],)).fetchone()
            quote = item.get("content", "")
            checks.append(bool(row and quote and item.get("url") and item["url"] == row[1]
                               and row[0][item.get("char_start", 0):item.get("char_end", 0)] == quote))
    for key in METRICS:
        if metrics[key] is not None:
            reasons[key] = None
    return metrics, {"missing_reasons": reasons, "human_gold_available": approved_gold,
                     "citation_integrity": sum(checks) / len(checks) if checks else None,
                     "citation_integrity_count": len(checks),
                     "unknown_author_count": sum(not d.get("author_hash") for d in documents[:8])}


def evaluate_retrieval(retriever: HybridRetriever, benchmark: dict | Path | str | None = None,
                       *, modes: tuple[str, ...] = RETRIEVAL_MODES,
                       limit: int = 8) -> dict[str, Any]:
    """Run all questions/modes; return JSON-safe predictions and nullable metrics.

    Per-question failures are retained as errors and never averaged as zero.
    This function writes nothing and never changes an experiment's approval.
    """
    if not isinstance(benchmark, dict):
        benchmark = load_benchmark(benchmark)
    _validate(benchmark)
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or any(m not in RETRIEVAL_MODES for m in modes):
        raise ValueError("modes must be unique supported retrieval modes")
    encoded = json.dumps(benchmark, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    results = []
    for question in benchmark["questions"]:
        for mode in modes:
            row = {"question_id": question["id"], "query": question["query"],
                   "category": question.get("category"), "mode": mode,
                   "human_review": question.get("human_review")}
            try:
                prediction = retriever.search(question["query"], focus=question.get("focus"), limit=limit, mode=mode)
                metrics, diagnostics = _metrics(question, prediction, retriever.database)
                row.update(result=prediction, metrics=metrics, diagnostics=diagnostics, error=None)
            except Exception as error:
                row.update(result=None, metrics=dict.fromkeys(METRICS),
                           diagnostics={"missing_reasons": {m: "retrieval_failed" for m in METRICS}},
                           error=type(error).__name__)
            results.append(row)
    summary = {}
    for mode in modes:
        rows = [r for r in results if r["mode"] == mode]
        summary[mode] = {"questions": len(rows), "errors": sum(r["error"] is not None for r in rows), "metrics": {}}
        for metric in METRICS:
            values = [r["metrics"][metric] for r in rows if r["metrics"][metric] is not None]
            summary[mode]["metrics"][metric] = {"mean": sum(values) / len(values) if values else None,
                                              "evaluated": len(values), "missing": len(rows) - len(values)}
    return {"evaluation_version": "retrieval-eval-v1", "benchmark_version": benchmark["version"],
            "benchmark_sha256": hashlib.sha256(encoded).hexdigest(), "benchmark": benchmark,
            "created_at": datetime.now(UTC).isoformat(), "modes": list(modes), "limit": limit,
            "experimental": True, "human_approved": False, "deployment_decision": "not-approved",
            "metric_definitions": {"scope_accuracy": "top-1 in human acceptable scope IDs, all annotated dimensions",
                "evidence_recall@8": "unique retrieved document IDs in first 8 / all human relevant IDs",
                "citation_precision": "human positive labels / retrieved unique first-8 documents; null unless fully labeled",
                "contradiction_coverage": "human dissent IDs retrieved in first 8 / all human dissent IDs",
                "author_diversity": "distinct known authors / known-author excerpts in first 8",
                "citation_integrity": "diagnostic only: verbatim offsets and URL match source DB"},
            "results": results, "summary": summary}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _blind_answer(result: dict | None) -> dict:
    """Explicit allowlist, never recursively dump a scored retrieval result."""
    if result is None:
        return {"candidates": [], "citations": [], "unavailable": True}
    candidates = []
    for subject in result.get("subjects", []):
        candidates.append({"subject_entity_id": subject["subject_entity_id"],
                           "subject_name": subject.get("subject_name", ""),
                           "episodes": [{k: ep.get(k) for k in (
                               "episode_entity_id", "episode_sort", "episode_name", "matching_description")}
                                        for ep in subject.get("episodes", [])]})
    citations = [{k: d.get(k) for k in ("document_id", "title", "content", "url",
                  "author_hash", "subject_entity_id", "episode_entity_id", "char_start", "char_end")}
                 for d in result.get("documents", [])]
    # Keep informational differences between experiments without their mode
    # names or implementation details (pack labels can reveal the experiment).
    pack = result.get("evidence_pack") or {}
    facts = [{k: item.get(k) for k in ("entity_id", "text", "fields", "fact_kind", "value", "url", "author_hash")}
             for item in pack.get("official-facts", [])]
    summaries = [{"text": item.get("summary", ""), "citable": False}
                 for item in pack.get("report_signposts", [])]
    return {"candidates": candidates, "citations": citations, "facts": facts,
            "summaries": summaries, "unavailable": False}


def prepare_blind_review(report: dict, seed: int = 0) -> dict[str, Any]:
    """Return {'blind': public worksheet, 'private': identity/mode sidecar}.

    Callers must store/share the two separately. Randomization is reproducible,
    not cryptographic secrecy; the seed and original report remain private.
    Ratings are 1..5 integers. Only the human supplies reviewer and ratings.
    """
    rng = random.Random(seed)
    grouped = {}
    for row in report["results"]:
        grouped.setdefault(row["question_id"], []).append(row)
    review_id = _digest({"benchmark": report["benchmark_sha256"], "seed": seed,
                         "results": report["results"]})
    blind = {"schema_version": "retrieval-blind-v1", "review_id": review_id,
             "reviewer": None, "rating_scale": "1 (poor) .. 5 (excellent)", "questions": []}
    private = {"schema_version": "retrieval-blind-private-v1", "review_id": review_id,
               "benchmark_sha256": report["benchmark_sha256"], "seed": seed, "cases": {}}
    for question_id, rows in grouped.items():
        rows = list(rows)
        if len({r["mode"] for r in rows}) != len(rows) or len({r["query"] for r in rows}) != 1:
            raise ValueError("duplicate modes or conflicting question identity")
        rng.shuffle(rows)
        question = {"question_id": question_id, "query": rows[0]["query"], "answers": []}
        for row in rows:
            case_id = f"case-{rng.getrandbits(128):032x}"
            answer = {"case_id": case_id, "answer": _blind_answer(row.get("result")),
                      "ratings": {"relevance": None, "citation_support": None, "usefulness": None},
                      "notes": None}
            identity = {"question_id": question_id, "query": question["query"],
                        "case_id": case_id, "answer": answer["answer"]}
            private["cases"][case_id] = {"question_id": question_id, "mode": row["mode"],
                                         "identity_sha256": _digest(identity)}
            question["answers"].append(answer)
        blind["questions"].append(question)
    blind["private_identity_sha256"] = _digest(private)
    return {"blind": blind, "private": private}


def score_blind_review(blind: dict, private: dict) -> dict[str, Any]:
    """Validate unmoved answers and aggregate actual human 1..5 ratings only.

    Blank/partial worksheets remain incomplete. Even complete worksheets never
    approve deployment; adoption requires a separate explicit human decision.
    """
    if (blind.get("schema_version") != "retrieval-blind-v1" or
            private.get("schema_version") != "retrieval-blind-private-v1" or
            not blind.get("review_id") or blind["review_id"] != private.get("review_id") or
            blind.get("private_identity_sha256") != _digest(private)):
        raise ValueError("blind/private review identity mismatch")
    records, seen = [], set()
    for question in blind["questions"]:
        for answer in question["answers"]:
            key = answer["case_id"]
            mapping = private["cases"].get(key)
            identity = {"question_id": question["question_id"], "query": question["query"],
                        "case_id": key, "answer": answer["answer"]}
            if key in seen or mapping is None or _digest(identity) != mapping["identity_sha256"]:
                raise ValueError("blind answer was duplicated, changed, or assigned to another question")
            seen.add(key)
            ratings = answer.get("ratings", {})
            if set(ratings) != {"relevance", "citation_support", "usefulness"}:
                raise ValueError("invalid rating fields")
            if any(value is not None and (type(value) is not int or not 1 <= value <= 5)
                   for value in ratings.values()):
                raise ValueError("human ratings must be null or integers in [1, 5]")
            records.append({"question_id": question["question_id"], "case_id": key,
                            "mode": mapping["mode"], "ratings": dict(ratings), "notes": answer.get("notes")})
    if seen != set(private["cases"]):
        raise ValueError("blind worksheet has missing cases")
    reviewer = blind.get("reviewer")
    has_ratings = any(v is not None for r in records for v in r["ratings"].values())
    if has_ratings and (not isinstance(reviewer, str) or not reviewer.strip()):
        raise ValueError("human reviewer identity is required for ratings")
    summary = {}
    for mode in sorted({r["mode"] for r in records}):
        rows = [r for r in records if r["mode"] == mode]
        summary[mode] = {}
        for field in ("relevance", "citation_support", "usefulness"):
            values = [r["ratings"][field] for r in rows if r["ratings"][field] is not None]
            summary[mode][field] = {"mean": sum(values) / len(values) if values else None,
                                    "rated": len(values), "missing": len(rows) - len(values)}
    return {"review_id": blind["review_id"], "reviewer": reviewer,
            "complete": bool(records) and all(v is not None for r in records for v in r["ratings"].values()),
            "human_approved": False, "deployment_decision": "requires-explicit-human-approval",
            "results": records, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--modes", nargs="+", choices=RETRIEVAL_MODES, default=list(RETRIEVAL_MODES))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", type=Path, help="New report path (never overwrites an existing file)")
    parser.add_argument("--semantic", action="store_true", help="Explicitly allow embedding requests")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    if args.semantic and args.root is None:
        parser.error("--semantic requires --root")
    if args.output is not None and args.output.exists():
        parser.error("--output already exists; choose a new report path")
    semantic = HybridSemanticSearch(args.root, args.database) if args.semantic else None
    try:
        report = evaluate_retrieval(HybridRetriever(args.database, semantic=semantic), args.benchmark,
                                    modes=tuple(args.modes), limit=args.limit)
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output is None:
            print(text)
        else:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(text + "\n")
        return 1 if any(row["error"] for row in report["results"]) else 0
    finally:
        if semantic is not None:
            semantic.close()


if __name__ == "__main__":
    raise SystemExit(main())

import copy
import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mikirin.retrieval_evaluation import (evaluate_retrieval, load_benchmark,
    main, prepare_blind_review, score_blind_review)


class OfflineRetriever:
    def __init__(self, database):
        self.database = database
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append(kwargs)
        return {"mode": kwargs["mode"], "trace": ["PRIVATE_ROUTE"],
                "provenance": {"private": "PRIVATE_FINGERPRINT"},
                "subjects": [{"subject_entity_id": "s1", "subject_name": "合成作品",
                              "score": .8, "episodes": [{"episode_entity_id": "e1", "episode_name": "合成章节", "score": .7}]}],
                "episodes": [{"episode_entity_id": "e1", "score": .7}],
                "documents": [{"document_id": "d1", "content": "连续原文", "url": "https://example.test/1",
                               "author_hash": "a1", "char_start": 0, "char_end": 4, "lexical_score": .4}],
                "evidence_pack": None}


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "synthetic.db"
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("CREATE TABLE source_document(id TEXT, content TEXT, url TEXT)")
            db.execute("INSERT INTO source_document VALUES ('d1','连续原文','https://example.test/1')")
        self.retriever = OfflineRetriever(self.database)
        self.benchmark = {"version": "synthetic-v1", "questions": [
            {"id": "q1", "query": "合成作品的合成问题", "category": "test", "gold": None,
             "annotation": None, "human_review": None}]}

    def test_default_resource_is_versioned_sixty_specific_unlabeled_questions(self):
        benchmark = load_benchmark()
        self.assertEqual(len(benchmark["questions"]), 60)
        self.assertEqual(len({q["id"] for q in benchmark["questions"]}), 60)
        self.assertEqual(len({q["category"] for q in benchmark["questions"]}), 9)
        self.assertTrue(benchmark["version"])
        for q in benchmark["questions"]:
            self.assertTrue(all(v is None for v in q["gold"].values()))
            self.assertIsNone(q["human_review"])
            self.assertIsNone(q["annotation"])
            self.assertNotIn("TODO", q["query"])

    def test_four_modes_all_questions_null_gold_and_real_source_fidelity(self):
        report = evaluate_retrieval(self.retriever, self.benchmark)
        self.assertEqual(len(report["results"]), 4)
        self.assertEqual(len(self.retriever.calls), 4)
        for row in report["results"]:
            self.assertIsNone(row["metrics"]["scope_accuracy"])
            self.assertIsNone(row["metrics"]["citation_precision"])
            self.assertEqual(row["metrics"]["author_diversity"], 1.)
            self.assertEqual(row["diagnostics"]["citation_integrity"], 1.)
            self.assertEqual(row["diagnostics"]["missing_reasons"]["scope_accuracy"], "human_gold_missing")
        self.assertFalse(report["human_approved"])
        self.assertEqual(report["summary"]["hybrid"]["metrics"]["scope_accuracy"]["evaluated"], 0)

    def test_synthetic_human_annotation_fixture_metrics_not_auto_gold(self):
        question = self.benchmark["questions"][0]
        question["gold"] = {"scope": {"subject_entity_ids": ["s1"], "episode_entity_ids": ["e1"]},
                            "evidence_document_ids": ["d1", "d2"],
                            "dissenting_document_ids": ["d2"], "citation_labels": {"d1": True}}
        report = evaluate_retrieval(self.retriever, self.benchmark, modes=("fts",))
        self.assertIsNone(report["results"][0]["metrics"]["scope_accuracy"])
        question["annotation"] = {"source": "human", "annotator": "synthetic-unit-test-fixture-not-real-review"}
        report = evaluate_retrieval(self.retriever, self.benchmark, modes=("fts",))
        self.assertEqual(report["results"][0]["metrics"], {
            "scope_accuracy": 1., "evidence_recall@8": .5, "citation_precision": 1.,
            "contradiction_coverage": 0., "author_diversity": 1.})
        self.assertFalse(report["human_approved"])

    def test_model_annotation_cannot_be_human_gold(self):
        question = self.benchmark["questions"][0]
        question.update(gold={"evidence_document_ids": ["d1"]}, annotation={"source": "model", "annotator": "model"})
        report = evaluate_retrieval(self.retriever, self.benchmark, modes=("fts",))
        self.assertIsNone(report["results"][0]["metrics"]["evidence_recall@8"])

    def test_failing_queries_are_not_scored_zero_and_error_text_is_redacted(self):
        with patch.object(self.retriever, "search", side_effect=RuntimeError("API_SECRET raw response")):
            report = evaluate_retrieval(self.retriever, self.benchmark)
        self.assertNotIn("API_SECRET", json.dumps(report))
        self.assertEqual(report["summary"]["hybrid"]["errors"], 1)
        self.assertIsNone(report["summary"]["hybrid"]["metrics"]["scope_accuracy"]["mean"])

    def test_benchmark_identity_and_validation(self):
        first = evaluate_retrieval(self.retriever, self.benchmark)
        second = evaluate_retrieval(self.retriever, self.benchmark)
        self.assertEqual(first["benchmark_sha256"], second["benchmark_sha256"])
        self.benchmark["questions"][0]["query"] += "修改"
        changed = evaluate_retrieval(self.retriever, self.benchmark)
        self.assertNotEqual(first["benchmark_sha256"], changed["benchmark_sha256"])
        self.benchmark["questions"] *= 2
        with self.assertRaises(ValueError):
            evaluate_retrieval(self.retriever, self.benchmark)

    def test_sixty_by_four_runner(self):
        report = evaluate_retrieval(self.retriever)
        self.assertEqual(len(report["results"]), 240)
        self.assertEqual(sum(s["errors"] for s in report["summary"].values()), 0)

    def test_blind_is_deterministic_anonymous_and_unscored(self):
        report = evaluate_retrieval(self.retriever, self.benchmark)
        prepared = prepare_blind_review(report, seed=0)
        self.assertEqual(prepared, prepare_blind_review(report, seed=0))
        text = json.dumps(prepared["blind"])
        for secret in ("PRIVATE_ROUTE", "PRIVATE_FINGERPRINT", "lexical_score", '"mode"', '"score"', "hybrid", "anchored"):
            self.assertNotIn(secret, text)
        scored = score_blind_review(**prepared)
        self.assertFalse(scored["complete"])
        self.assertFalse(scored["human_approved"])
        self.assertIsNone(scored["summary"]["fts"]["relevance"]["mean"])

    def test_blind_detects_swaps_edits_duplicates_and_missing_cases(self):
        original = prepare_blind_review(evaluate_retrieval(self.retriever, self.benchmark))
        for mutation in ("answer", "query", "duplicate", "missing", "private", "mapping"):
            prepared = copy.deepcopy(original)
            question = prepared["blind"]["questions"][0]
            if mutation == "answer":
                question["answers"][0]["answer"]["citations"][0]["content"] = "forged"
            elif mutation == "query":
                question["query"] += "changed"
            elif mutation == "duplicate":
                question["answers"].append(copy.deepcopy(question["answers"][0]))
            elif mutation == "missing":
                question["answers"].pop()
            elif mutation == "private":
                prepared["private"]["review_id"] = "another-review"
            else:
                next(iter(prepared["private"]["cases"].values()))["mode"] = "tampered"
            with self.assertRaises(ValueError, msg=mutation):
                score_blind_review(**prepared)

    def test_blind_actual_fixture_scores_require_reviewer_never_approve(self):
        prepared = prepare_blind_review(evaluate_retrieval(self.retriever, self.benchmark))
        for answer in prepared["blind"]["questions"][0]["answers"]:
            answer["ratings"] = {"relevance": 4, "citation_support": 3, "usefulness": 5}
        with self.assertRaises(ValueError):
            score_blind_review(**prepared)
        prepared["blind"]["reviewer"] = "synthetic-test-fixture"
        scored = score_blind_review(**prepared)
        self.assertTrue(scored["complete"])
        self.assertFalse(scored["human_approved"])
        self.assertEqual(scored["summary"]["fts"]["relevance"]["mean"], 4.)
        prepared["blind"]["questions"][0]["answers"][0]["ratings"]["relevance"] = True
        with self.assertRaises(ValueError):
            score_blind_review(**prepared)

    def test_standalone_cli_stdout_and_no_overwrite(self):
        with patch("mikirin.retrieval_evaluation.HybridRetriever", return_value=self.retriever), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--database", str(self.database), "--modes", "fts"]), 0)
        self.assertEqual(len(json.loads(output.getvalue())["results"]), 60)
        with self.assertRaises(SystemExit):
            main(["--database", str(self.database), "--output", str(self.database)])


if __name__ == "__main__":
    unittest.main()

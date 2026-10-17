"""Synthetic fixtures only: these tests are not real human quality acceptance."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from contextlib import closing

from mikirin.p1_evaluation import (
    calibrate_slang_thresholds, evaluate_graph_ratings, evaluate_implicit_review,
    evaluate_slang_golden, load_graph_queries, prepare_implicit_review,
    prepare_monthly_alias_review, run_graph_comparison,
)


class P1EvaluationTests(unittest.TestCase):
    def _graph_db(self):
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)
        db.executescript("""
            CREATE TABLE entity(id TEXT,canonical_name TEXT);
            INSERT INTO entity VALUES('hit','可读作品名');
            CREATE TABLE derived_edge(src_entity_id TEXT,dst_entity_id TEXT,kind TEXT,
                                      weight REAL,evidence_json TEXT,build_id TEXT);
            INSERT INTO derived_edge VALUES('seed','hit','tag_similarity',1,'{}','b');
            CREATE TABLE node_pagerank(node_entity_id TEXT,graph_kind TEXT,score REAL,build_id TEXT);
            INSERT INTO node_pagerank VALUES('hit','similar_work',1,'b');
        """)
        return db

    def _queries(self):
        return [{"query_id": f"q{i}", "query": f"真实提问{i}", "seed_entity_ids": ["seed"],
                 "task": "similar_work", "lexical_candidates": [{"entity_id": "hit", "score": 1,
                   "document_id": "doc", "quote": "逐字原文", "attribution": "一位作者的说法"}],
                 "vector_candidates": [{"entity_id": "hit", "score": 0.8}],
                 "expected_entity_ids": None} for i in range(30)]

    def test_packaged_queries_have_no_invented_gold(self):
        rows = load_graph_queries()
        self.assertEqual(len(rows), 30)
        self.assertTrue(all(row["expected_entity_ids"] is None for row in rows))
        self.assertEqual(len({row["query"] for row in rows}), 30)

    def test_comparison_is_deterministic_blind_readable_and_unrated(self):
        db = self._graph_db()
        before = db.total_changes
        result = run_graph_comparison(db, self._queries(), build_id="b", seed=9)
        self.assertEqual(result, run_graph_comparison(db, self._queries(), build_id="b", seed=9))
        self.assertEqual(db.total_changes, before)
        blind = result["blind"]
        serialized = json.dumps(blind, ensure_ascii=False)
        self.assertIn("可读作品名", serialized)
        self.assertIn("逐字原文", serialized)
        self.assertIn("一位作者的说法", serialized)
        self.assertNotIn("graph_score", serialized)
        self.assertNotIn("fusion", serialized)
        for row in blind["queries"]:
            self.assertTrue(all(arm["relevance"] is None for arm in row["arms"].values()))
        self.assertIsNone(result["metrics"]["modes"]["simple"]["top1_accuracy"])
        self.assertFalse(result["metrics"]["default_change_authorized"])
        review = evaluate_graph_ratings(blind, result["private"])
        self.assertEqual(review["status"], "awaiting_human_labels")
        # Labels below are a synthetic test fixture, not actual user ratings.
        for row in blind["queries"]:
            for arm in row["arms"].values():
                arm.update(relevance=4, surprise=3)
        review = evaluate_graph_ratings(blind, result["private"])
        self.assertEqual(review["modes"]["graph"]["relevance"], {"count": 30, "mean": 4})
        self.assertFalse(review["default_change_authorized"])

    def test_output_refuses_overwrite_and_missing_vectors_are_explicit(self):
        rows = self._queries()
        del rows[0]["vector_candidates"]
        with tempfile.TemporaryDirectory() as temporary:
            result = run_graph_comparison(self._graph_db(), rows, build_id="b", output_dir=temporary)
            self.assertFalse(result["metrics"]["vector_inputs_complete"])
            path = Path(temporary) / "graph-blind.json"
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                run_graph_comparison(self._graph_db(), rows, build_id="b", output_dir=temporary)
            self.assertEqual(path.read_bytes(), original)

    def test_comparison_validates_count_and_ids(self):
        with self.assertRaises(ValueError):
            run_graph_comparison(self._graph_db(), [], build_id="b")
        rows = self._queries()
        rows[1]["query_id"] = rows[0]["query_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            run_graph_comparison(self._graph_db(), rows, build_id="b")

    def test_calibration_counts_labels_not_sampling_strata(self):
        rows = [{"candidate_id": str(i), "score": i / 200,
                 "sampling_stratum": "positive" if i < 100 else "negative", "label": ""}
                for i in range(200)]
        report = calibrate_slang_thresholds(rows, threshold_version="test")
        self.assertEqual(report["positive_count"], 0)
        self.assertIsNone(report["thresholds"])
        for i, row in enumerate(rows):
            row["label"] = "positive" if i >= 100 else "negative"
        report = calibrate_slang_thresholds(rows, threshold_version="test")
        self.assertEqual(report["thresholds"]["candidate_score_threshold"], 0.5)
        self.assertIsNone(report["thresholds"]["single_char_score_threshold"])
        self.assertIsNone(report["held_out_acceptance"])

    def test_calibration_requires_both_classes_and_rejects_nan(self):
        rows = [{"candidate_id": str(i), "score": 0.9, "label": "positive"} for i in range(200)]
        self.assertEqual(calibrate_slang_thresholds(rows, threshold_version="t")["status"], "awaiting_human_labels")
        rows[0]["score"] = float("nan")
        with self.assertRaises(ValueError):
            calibrate_slang_thresholds(rows, threshold_version="t")

    def test_golden_does_not_treat_blank_as_abstention(self):
        rows = [{"query_id": str(i), "query": "锯" if i == 0 else f"测试{i}", "expected": ""} for i in range(50)]
        report = evaluate_slang_golden(rows, search=lambda query: ["chainsaw"])
        self.assertIsNone(report["top1_accuracy"])
        self.assertFalse(report["acceptance_passed"])
        for row in rows:
            row["expected"] = '["chainsaw"]'
        report = evaluate_slang_golden(rows, search=lambda query: ["chainsaw"])
        self.assertEqual(report["top1_accuracy"], 1)
        self.assertTrue(report["acceptance_passed"])
        report = evaluate_slang_golden(rows, search=lambda query: ["wrong"])
        self.assertEqual(report["wrong_hits"], 50)
        self.assertFalse(report["acceptance_passed"])
        report = evaluate_slang_golden(rows, search=lambda query: ["chainsaw", "wrong"])
        self.assertEqual(report["top1_accuracy"], 1)
        self.assertEqual(report["wrong_hits"], 50)
        self.assertFalse(report["acceptance_passed"])

    def test_no_single_character_hits_cannot_pass_precision_gate(self):
        rows = [{"query_id": str(i), "query": f"团长{i}", "expected": ["haruhi"]} for i in range(50)]
        result = evaluate_slang_golden(rows, search=lambda query: ["haruhi"])
        self.assertIsNone(result["single_char_precision"])
        self.assertFalse(result["acceptance_passed"])

    def test_implicit_quote_verification_is_separate_from_human_precision(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.db"
            with closing(sqlite3.connect(path)) as db, db:
                db.executescript("""
                    CREATE TABLE implicit_edge(src_entity_id TEXT,dst_entity_id TEXT,relation TEXT,
                      sentiment TEXT,provenance_json TEXT,build_id TEXT);
                    CREATE TABLE implicit_edge_evidence(id TEXT,document_id TEXT,claim_quote TEXT,
                      char_start INT,char_end INT,author_hash TEXT,run_id TEXT,src_entity_id TEXT,
                      dst_entity_id TEXT,relation TEXT,sentiment TEXT);
                    CREATE TABLE source_document(id TEXT,content TEXT);
                    INSERT INTO source_document VALUES('d','甲与乙是朋友');
                    INSERT INTO implicit_edge VALUES('a','b','friend','neutral','{"run_id":"r"}','b');
                    INSERT INTO implicit_edge_evidence VALUES('e','d','甲与乙',0,3,'author','r','a','b','friend','neutral');
                """)
            rows = prepare_implicit_review(path, build_id="b")
            self.assertTrue(rows[0]["quote_verified"])
            report = evaluate_implicit_review(rows)
            self.assertIsNone(report["human_precision"])
            self.assertFalse(report["acceptance_passed"])
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("UPDATE source_document SET content='甲和乙是朋友'")
            self.assertFalse(prepare_implicit_review(path, build_id="b")[0]["quote_verified"])

    def test_implicit_gate_needs_100_labeled_and_all_quotes(self):
        rows = [{"review_id": str(i), "label": "correct", "quote_verified": True} for i in range(100)]
        self.assertTrue(evaluate_implicit_review(rows)["acceptance_passed"])
        self.assertFalse(evaluate_implicit_review(rows[:99])["acceptance_passed"])
        rows[0]["quote_verified"] = False
        self.assertFalse(evaluate_implicit_review(rows)["acceptance_passed"])

    def test_monthly_sample_is_distinct_reproducible_and_blank(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "aliases.db"
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("CREATE TABLE entity_alias(id TEXT,source_kind TEXT,status TEXT)")
                db.executemany("INSERT INTO entity_alias VALUES(?,'mined','confirmed')", [(str(i),) for i in range(60)])
                db.execute("INSERT INTO entity_alias VALUES('bad','mined','rejected')")
            first = prepare_monthly_alias_review(path, month="2026-10")
            self.assertEqual(first, prepare_monthly_alias_review(path, month="2026-10"))
            self.assertEqual(len(first), 50)
            self.assertNotIn("bad", {row["id"] for row in first})
            self.assertTrue(all(row["label"] is None for row in first))


if __name__ == "__main__":
    unittest.main()

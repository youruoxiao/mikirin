from __future__ import annotations

import csv
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from mikirin.schema import CONTENT_SCHEMA
from scripts.prepare_p1_manual_review import PreparationError, prepare_p1_manual_review


class PrepareP1ManualReviewTests(unittest.TestCase):
    def _database(self, directory: str, *, complete: bool = True) -> Path:
        database = Path(directory) / "side.db"
        with closing(sqlite3.connect(database)) as db, db:
            db.executescript(CONTENT_SCHEMA)
            db.execute("INSERT INTO schema_metadata VALUES('schema_version','10')")
            db.execute(
                "INSERT INTO build(id,started_at,code_version,status) VALUES(?,?,?,'building')",
                ("build-1", "now", "test"),
            )
            db.execute(
                "INSERT INTO ingestion_run VALUES(?,?,?,?,?,?,?,?,?)",
                ("ingest-1", "test", "now", None, "v", "v", "done", 0, 0),
            )
            db.execute(
                "INSERT INTO source_asset(id,ingestion_run_id,local_path,content_hash,fetched_at) "
                "VALUES('asset-1','ingest-1','fixture','hash','now')"
            )
            for index in range(35):
                entity_id = f"subject:{index:03d}"
                db.execute(
                    "INSERT INTO entity VALUES(?,?,?,'active')",
                    (entity_id, "subject", f"Work {index:03d}"),
                )
                db.execute(
                    "INSERT INTO subjects(entity_id,type,name,name_cn,source_asset_id) "
                    "VALUES(?,2,?,?,?)",
                    (entity_id, f"Work {index:03d}", f"作品{index:03d}", "asset-1"),
                )
            # Insert graph rows only after every referenced subject exists.
            for index in range(35):
                db.execute(
                    "INSERT INTO task_pagerank VALUES(?,?,?,?,?,?,?)",
                    (
                        f"subject:{index:03d}",
                        f"subject:{(index + 1) % 35:03d}",
                        "similar_work",
                        1,
                        1.0,
                        "{}",
                        "build-1",
                    ),
                )
            db.execute(
                "INSERT INTO slang_run VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "run-1",
                    "build-1",
                    "test",
                    "calibrated-v1",
                    "watermark",
                    "input",
                    "output",
                    "{}",
                    "now",
                    "now",
                    "completed" if complete else "running",
                ),
            )
            for index in range(200):
                db.execute(
                    "INSERT INTO slang_candidate VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        f"candidate:{index:03d}",
                        "run-1",
                        f"黑话{index:03d}",
                        f"黑话{index:03d}",
                        "cross_work",
                        10,
                        5,
                        3,
                        "{}",
                        float(200 - index),
                        0.5,
                        "shortlisted",
                    ),
                )
            pairs = []
            for left in range(35):
                for right in range(left + 1, 35):
                    pairs.append((left, right))
            for index, (left, right) in enumerate(pairs[:100]):
                db.execute(
                    "INSERT INTO implicit_edge VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        f"subject:{left:03d}",
                        f"subject:{right:03d}",
                        "friend",
                        "positive",
                        2,
                        1,
                        0.8,
                        "{}",
                        "build-1",
                    ),
                )
        return database

    def test_generates_blank_deterministic_packages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            first_dir = Path(directory) / "first"
            second_dir = Path(directory) / "second"
            first = prepare_p1_manual_review(database, "build-1", "run-1", first_dir)
            second = prepare_p1_manual_review(database, "build-1", "run-1", second_dir)

            for key in first:
                self.assertEqual(first[key].read_bytes(), second[key].read_bytes())

            with first["a4_calibration"].open(newline="", encoding="utf-8") as handle:
                calibration = list(csv.DictReader(handle))
            self.assertEqual(len(calibration), 200)
            self.assertEqual(
                {row["sampling_stratum"] for row in calibration}, {"high_score", "low_score"}
            )
            self.assertTrue(all(row["label"] == "" for row in calibration))

            with first["a4_golden_queries"].open(newline="", encoding="utf-8") as handle:
                golden = list(csv.DictReader(handle))
            self.assertEqual(len(golden), 50)
            self.assertTrue(all(row["expected"] == "" for row in golden))

            with first["a5_implicit_edge_review"].open(newline="", encoding="utf-8") as handle:
                edges = list(csv.DictReader(handle))
            self.assertEqual(len(edges), 100)
            self.assertTrue(all(row["label"] == "" for row in edges))

            package = json.loads(first["a5_query_ab_json"].read_text(encoding="utf-8"))
            self.assertEqual(package["arms"], {"a": "simple", "b": "fusion"})
            self.assertEqual(len(package["queries"]), 30)
            self.assertNotIn("rating", json.dumps(package))
            self.assertNotIn("user_id", json.dumps(package))
            with first["a5_query_ab_csv"].open(newline="", encoding="utf-8") as handle:
                ab_rows = list(csv.DictReader(handle))
            self.assertEqual(len(ab_rows), 30)
            self.assertTrue(all(row["human_preference"] == "" for row in ab_rows))

    def test_fails_before_writing_when_run_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory, complete=False)
            output = Path(directory) / "out"
            with self.assertRaisesRegex(PreparationError, "not complete"):
                prepare_p1_manual_review(database, "build-1", "run-1", output)
            self.assertFalse(output.exists())

    def test_fails_clearly_when_a_required_stratum_is_short(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("DELETE FROM slang_candidate WHERE id LIKE 'candidate:19%'")
            with self.assertRaisesRegex(PreparationError, "A4 calibration needs 200"):
                prepare_p1_manual_review(database, "build-1", "run-1", Path(directory) / "out")


if __name__ == "__main__":
    unittest.main()

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from mikirin.memory_review import export_weekly_review, review_report
from mikirin.settlement import MemorySettlement
from mikirin.user_store import UserStore


class MemoryReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / "user.db"
        self.output = self.root / "review.json"
        UserStore(self.db)

    def seed(self, count=25):
        with closing(sqlite3.connect(self.db)) as db, db:
            for index in range(count):
                db.execute("INSERT INTO memory_note(id, content, source_session_id, valid_from, "
                           "created_at) VALUES(?,?,'s','2020-01-01','2020-01-01')",
                           (f"note:{index}", f"测试偏好 {index}"))
            db.execute("INSERT INTO memory_note(id, content, valid_from, valid_until, created_at) "
                       "VALUES('expired','失效事实','2020-01-01','2021-01-01','2020-01-01')")

    def export(self):
        return export_weekly_review(self.db, self.output, week="2026-W42")

    def label(self, correct, *, labeled=20):
        payload = json.loads(self.output.read_text(encoding="utf-8"))
        for index, card in enumerate(payload["cards"]):
            card["label"] = ("correct" if index < correct else "incorrect") if index < labeled else ""
        self.output.write_text(json.dumps(payload), encoding="utf-8")

    def test_deterministic_twenty_blank_labels_readonly_database_and_source_evidence(self):
        self.seed()
        MemorySettlement(self.db).enqueue("s", [{"role": "user", "content": "原始偏好证据"}])
        before = self.db.read_bytes()
        result = self.export()
        other = self.root / "same-week.json"
        export_weekly_review(self.db, other, week="2026-W42")
        self.assertEqual(self.output.read_bytes(), other.read_bytes())
        self.assertEqual(self.db.read_bytes(), before)
        self.assertEqual(len(result["cards"]), 20)
        self.assertTrue(all(card["label"] == "" for card in result["cards"]))
        self.assertNotIn("expired", [card["note_id"] for card in result["cards"]])
        self.assertEqual(result["cards"][0]["settlement_inputs"][0]["messages"][0]["content"],
                         "原始偏好证据")
        report = review_report(self.output)
        self.assertIsNone(report["accuracy"])
        self.assertFalse(report["passed"])
        self.assertFalse(report["upgrade_recommended"])

    def test_threshold_only_complete_and_strictly_below_ninety(self):
        self.seed()
        self.export()
        self.label(18)
        report = review_report(self.output)
        self.assertEqual(report["accuracy"], 0.9)
        self.assertTrue(report["passed"])
        self.assertFalse(report["upgrade_recommended"])
        self.label(17)
        self.assertTrue(review_report(self.output)["upgrade_recommended"])
        self.label(0, labeled=19)
        report = review_report(self.output)
        self.assertFalse(report["complete"])
        self.assertFalse(report["upgrade_recommended"])

    def test_fewer_than_twenty_cannot_pass(self):
        self.seed(3)
        self.export()
        self.label(3)
        report = review_report(self.output)
        self.assertEqual(report["accuracy"], 1)
        self.assertFalse(report["passed"])
        self.assertFalse(report["complete"])

    def test_reexport_preserves_human_labels_and_refuses_different_week(self):
        self.seed()
        self.export()
        self.label(18)
        before = self.output.read_bytes()
        self.export()
        self.assertEqual(self.output.read_bytes(), before)
        with self.assertRaises(ValueError):
            export_weekly_review(self.db, self.output, week="2026-W43")
        self.assertEqual(self.output.read_bytes(), before)

    def test_invalid_labels_or_edited_evidence_are_rejected(self):
        self.seed()
        original = self.export()
        for field, value in (("label", "auto-correct"), ("content", "改写证据")):
            payload = json.loads(json.dumps(original))
            payload["cards"][0][field] = value
            self.output.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                review_report(self.output)

    def test_invalid_week_rejected_before_database_open(self):
        for week in ("2026-42", "2026-W54", "2026-W00"):
            with self.assertRaises(ValueError):
                export_weekly_review(self.db, self.output, week=week)


if __name__ == "__main__":
    unittest.main()

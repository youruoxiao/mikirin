import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from mikirin.conversation_focus import EMPTY_FOCUS, ConversationFocus
from mikirin.topic import EMPTY_TOPIC, build_topic_layer


class TopicInsightsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.database = Path(self.tmp.name) / "topic.db"
        with closing(sqlite3.connect(self.database)) as db, db:
            db.executescript(
                """
                CREATE TABLE subjects (
                    entity_id TEXT PRIMARY KEY, name TEXT, name_cn TEXT,
                    date TEXT, platform TEXT
                );
                CREATE TABLE score_snapshot (
                    subject_entity_id TEXT, score REAL, rank INTEGER
                );
                CREATE TABLE episodes (
                    entity_id TEXT PRIMARY KEY, subject_entity_id TEXT,
                    sort REAL
                );
                CREATE TABLE community_document (
                    subject_entity_id TEXT, episode_entity_id TEXT,
                    source_kind TEXT
                );
                INSERT INTO subjects VALUES ('s1', 'Name', '作品', '2024', 'web');
                INSERT INTO score_snapshot VALUES ('s1', 8.8, 2);
                INSERT INTO episodes VALUES ('e1', 's1', 3.0);
                """
            )

    def tearDown(self):
        self.tmp.cleanup()

    def focus(self, **kwargs):
        return ConversationFocus(active_subject_id="s1", confidence=0.9, **kwargs)

    def test_old_database_keeps_original_card(self):
        result = build_topic_layer(self.database, self.focus())
        self.assertIn("subject:", result)
        self.assertNotIn("community-report", result)
        self.assertNotIn("discussion-themes", result)

    def test_report_is_injected_in_fixed_order(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute(
                """CREATE TABLE community_report (
                    subject_entity_id TEXT, reception_summary TEXT,
                    strengths_json TEXT, weaknesses_json TEXT,
                    discussion_points_json TEXT, key_episodes_json TEXT,
                    notable_disagreements_json TEXT
                )"""
            )
            db.execute(
                "INSERT INTO community_report VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("s1", "summary <x>", json.dumps([{"point": "strong & good"}]),
                 json.dumps([{"point": "weak"}]), json.dumps([{"point": "point"}]),
                 json.dumps([{"sort": 3, "reason": "episode"}]),
                 json.dumps([{"point": "disagree"}])),
            )
        result = build_topic_layer(self.database, self.focus())
        self.assertLess(result.index("reception_summary"), result.index("strengths"))
        self.assertIn("summary &lt;x&gt;", result)
        self.assertIn("strong &amp; good", result)

    def test_episode_theme_uses_sort_lookup_and_two_quotes(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute(
                """CREATE TABLE discussion_theme (
                    id TEXT, subject_entity_id TEXT, episode_entity_id TEXT,
                    theme_text TEXT, aspect_kind TEXT, quotes_json TEXT
                )"""
            )
            db.execute(
                "INSERT INTO discussion_theme VALUES (?, ?, ?, ?, ?, ?)",
                ("t1", "s1", "e1", "theme ```", "plot",
                 json.dumps([{"quote": "q1"}, {"quote": "q2"}, {"quote": "q3"}])),
            )
        result = build_topic_layer(self.database, self.focus(active_episode_sort=3.0))
        self.assertIn("theme_text:", result)
        self.assertIn("aspect_kind: plot", result)
        self.assertEqual(result.count("quote:"), 2)
        self.assertIn("specific claims use quotes", result)
        self.assertNotIn("```", result)

    def test_same_focus_is_byte_stable(self):
        focus = self.focus(active_episode_id="e1", active_episode_sort=3.0)
        self.assertEqual(build_topic_layer(self.database, focus), build_topic_layer(self.database, focus))

    def test_empty_focus_is_exact_constant(self):
        self.assertEqual(build_topic_layer(Path("missing.db"), EMPTY_FOCUS), EMPTY_TOPIC)


if __name__ == "__main__":
    unittest.main()

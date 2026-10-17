import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from mikirin.user_store import USER_SCHEMA, UserStore


class UserStoreTests(unittest.TestCase):
    def test_legacy_migration_is_idempotent_and_preserves_independent_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "user.db"
            with closing(sqlite3.connect(path)) as db, db:
                db.executescript(USER_SCHEMA.split("CREATE TABLE IF NOT EXISTS memory_schema_migration")[0])
                db.execute("CREATE TABLE rss_independent(id TEXT PRIMARY KEY)")
                db.execute("INSERT INTO rss_independent VALUES('keep')")
                db.execute("INSERT INTO memory_note(id, content, valid_from, created_at) "
                           "VALUES('legacy','已有的记忆卡片','2020-01-01','2020-01-01')")
                db.execute("CREATE TABLE memory_settlement_receipt(session_id TEXT, content_hash TEXT, "
                           "decision_json TEXT, note_id TEXT, created_at TEXT, "
                           "PRIMARY KEY(session_id, content_hash))")
            UserStore(path)
            store = UserStore(path)
            self.assertEqual(len(store.search_memory_notes("记忆卡片")), 1)
            with closing(sqlite3.connect(path)) as db:
                self.assertEqual(db.execute("SELECT id FROM rss_independent").fetchone()[0], "keep")
                self.assertEqual(db.execute("SELECT count(*) FROM memory_note_fts").fetchone()[0], 1)
                self.assertIn("source_job_id", [row[1] for row in db.execute(
                    "PRAGMA table_info(memory_settlement_receipt)")])

    def test_fts_triggers_follow_insert_update_delete_without_reopening_store(self):
        with tempfile.TemporaryDirectory() as directory:
            store = UserStore(Path(directory) / "user.db")
            with closing(sqlite3.connect(store.path)) as db, db:
                db.execute("INSERT INTO memory_note(id, content, valid_from, created_at) "
                           "VALUES('n','喜欢温柔的故事','2020-01-01','2020-01-01')")
            self.assertEqual(len(store.search_memory_notes("温柔")), 1)
            with closing(sqlite3.connect(store.path)) as db, db:
                db.execute("UPDATE memory_note SET content='喜欢热血的故事' WHERE id='n'")
            self.assertEqual(store.search_memory_notes("温柔的故事"), [])
            self.assertEqual(len(store.search_memory_notes("热血的故事")), 1)
            with closing(sqlite3.connect(store.path)) as db, db:
                db.execute("UPDATE memory_note SET valid_until='2021-01-01' WHERE id='n'")
            self.assertEqual(store.search_memory_notes("热血的故事"), [])
            self.assertEqual(len(store.search_memory_notes("热血的故事", include_expired=True)), 1)
            with closing(sqlite3.connect(store.path)) as db, db:
                db.execute("DELETE FROM memory_note WHERE id='n'")
                self.assertEqual(db.execute("SELECT count(*) FROM memory_note_fts").fetchone()[0], 0)

    def test_progress_is_authoritative_and_history_is_temporal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = UserStore(Path(directory) / "user.db")
            previous, first = store.update_progress(
                subject_entity_id="bgm:subject:328609",
                episode_entity_id="bgm:episode:1",
                progress_value=1,
                status="watching",
                source_message_id="m1",
                confidence=1.0,
                source_session_id="s1",
            )
            self.assertIsNone(previous)
            self.assertEqual(first.progress_value, 1)
            previous, current = store.update_progress(
                subject_entity_id="bgm:subject:328609",
                episode_entity_id="bgm:episode:7",
                progress_value=7,
                status="watching",
                source_message_id="m2",
                confidence=1.0,
                source_session_id="s1",
            )
            self.assertEqual(previous.progress_value, 1)
            self.assertEqual(store.progress("bgm:subject:328609").progress_value, 7)
            with closing(sqlite3.connect(store.path)) as db:
                rows = db.execute(
                    "SELECT valid_until FROM memory_note ORDER BY created_at, rowid"
                ).fetchall()
            self.assertIsNotNone(rows[0][0])
            self.assertIsNone(rows[1][0])

    def test_lower_confidence_cannot_silently_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = UserStore(Path(directory) / "user.db")
            store.update_progress(
                subject_entity_id="subject",
                episode_entity_id=None,
                progress_value=10,
                status="watching",
                source_message_id="confirmed",
                confidence=1.0,
            )
            with self.assertRaisesRegex(ValueError, "lower-confidence"):
                store.update_progress(
                    subject_entity_id="subject",
                    episode_entity_id=None,
                    progress_value=11,
                    status="watching",
                    source_message_id="inferred",
                    confidence=0.6,
                )
            self.assertEqual(store.progress("subject").progress_value, 10)

    def test_session_message_mirror_is_idempotent_and_searchable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = UserStore(Path(directory) / "user.db")
            store.register_session("s1", jsonl_path=Path("s1.jsonl"))
            inserted = store.mirror_message(
                session_id="s1",
                entry_id="e1",
                role="user",
                content="我最近在看孤独摇滚",
            )
            duplicate = store.mirror_message(
                session_id="s1",
                entry_id="e1",
                role="user",
                content="我最近在看孤独摇滚",
            )
            self.assertTrue(inserted)
            self.assertFalse(duplicate)
            hits = store.search_messages("孤独摇滚")
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["session_id"], "s1")
            context = store.surrounding_messages(hits[0]["message_id"], radius=2)
            self.assertEqual(context[0]["content"], "我最近在看孤独摇滚")
            self.assertEqual(store.recent_sessions(limit=1)[0]["session_id"], "s1")

    def test_progress_memory_card_is_discoverable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = UserStore(Path(directory) / "user.db")
            store.update_progress(
                subject_entity_id="subject",
                episode_entity_id="episode",
                progress_value=3,
                status="watching",
                source_message_id="m1",
                confidence=1,
            )
            notes = store.search_memory_notes("观看进度")
            self.assertEqual(len(notes), 1)
            self.assertIn("3", notes[0]["content"])


if __name__ == "__main__":
    unittest.main()

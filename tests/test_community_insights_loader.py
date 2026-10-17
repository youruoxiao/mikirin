import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from mikirin.layout import ProjectLayout
from mikirin.publish import Publisher
from mikirin.sources.community_insights_loader import CommunityInsightsLoader


class CommunityInsightsLoaderTests(unittest.TestCase):
    def _database(self, root: Path) -> Path:
        database = Publisher(ProjectLayout(root)).create_empty_build(
            build_id="build-test", code_version="test"
        )
        with closing(sqlite3.connect(database)) as db, db:
            asset = "asset:test"
            db.execute(
                "INSERT INTO ingestion_run(id, source, started_at, source_version, parser_version, status) "
                "VALUES (?, 'test', 'now', 'test', 'test', 'loaded')", ("run:test",)
            )
            db.execute(
                "INSERT INTO source_asset(id, ingestion_run_id, local_path, content_hash, fetched_at) "
                "VALUES (?, 'run:test', 'test', 'hash', 'now')", (asset,)
            )
            db.execute("INSERT INTO entity VALUES ('subject:1', 'subject', 'S', 'provisional')")
            db.execute("INSERT INTO subjects(entity_id, type, name, source_asset_id) VALUES ('subject:1', 1, 'S', ?)", (asset,))
            db.execute("INSERT INTO entity VALUES ('episode:1', 'episode', 'E', 'provisional')")
            db.execute("INSERT INTO episodes(entity_id, subject_entity_id, source_asset_id) VALUES ('episode:1', 'subject:1', ?)", (asset,))
        return database

    def test_load_is_idempotent_and_preserves_quotes_and_fts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = self._database(root)
            staging = root / "staging"
            staging.mkdir()
            brief = {"subject_entity_id": "subject:1", "reception_summary": "warm reception", "strengths": ["writing"], "coverage": {"comments": 2}, "source_digest": "sha256:x"}
            theme = {"theme": "visual rhythm", "aspect": "animation", "quotes": [{"comment_id": "c1", "quote": "the cuts breathe"}]}
            (staging / "briefs.jsonl").write_text(json.dumps(brief) + "\n", encoding="utf-8")
            (staging / "maps.jsonl").write_text(json.dumps({"subject_entity_id": "subject:1", "episode_entity_id": "episode:1", "themes": [theme], "prompt_version": "theme-v1"}) + "\n", encoding="utf-8")
            loader = CommunityInsightsLoader()
            self.assertEqual(loader.load_community_briefs(staging, database, "build-test").counts["loaded"], 1)
            first = loader.load_discussion_maps(staging, database, "build-test")
            second = loader.load_discussion_maps(staging, database, "build-test")
            self.assertEqual(first.counts["loaded"], 1)
            self.assertEqual(second.counts["loaded"], 1)
            with closing(sqlite3.connect(database)) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM discussion_theme").fetchone()[0], 1)
                quotes = db.execute("SELECT quotes_json FROM discussion_theme").fetchone()[0]
                self.assertEqual(json.loads(quotes)[0]["comment_id"], "c1")
                self.assertIsNotNone(db.execute("SELECT 1 FROM community_report_fts WHERE community_report_fts MATCH 'warm'").fetchone())
                self.assertIsNotNone(db.execute("SELECT 1 FROM discussion_theme_fts WHERE discussion_theme_fts MATCH 'rhythm'").fetchone())

    def test_dangling_subject_and_wrong_episode_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = self._database(root)
            staging = root / "staging"
            staging.mkdir()
            (staging / "briefs.jsonl").write_text(json.dumps({"subject_entity_id": "missing"}) + "\n", encoding="utf-8")
            (staging / "maps.jsonl").write_text(json.dumps({"subject_entity_id": "subject:1", "episode_entity_id": "missing", "themes": []}) + "\n", encoding="utf-8")
            loader = CommunityInsightsLoader()
            self.assertEqual(loader.load_community_briefs(staging, database, "build-test").counts["dangling"], 1)
            self.assertEqual(loader.load_discussion_maps(staging, database, "build-test").counts["dangling"], 1)


if __name__ == "__main__":
    unittest.main()

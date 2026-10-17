import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from mikirin.episode_card import CARD_SCHEMA_VERSION
from mikirin.schema import CONTENT_SCHEMA
from mikirin.vector_sources import (
    iter_vector_sources,
)


class VectorSourcesTests(unittest.TestCase):
    def test_scope_sorting_metadata_and_empty_filtering(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            database = Path(tmp) / "content.sqlite"
            with sqlite3.connect(database) as db:
                db.executescript(CONTENT_SCHEMA)
                db.execute("INSERT INTO ingestion_run VALUES ('r', 'test', 'now', NULL, '1', '1', 'done', 0, 0)")
                db.execute("INSERT INTO source_asset VALUES ('a', 'r', NULL, 'test', 'hash', 'now', NULL)")
                db.execute("INSERT INTO entity VALUES ('s1', 'subject', 'Show', 'active')")
                db.execute("INSERT INTO subjects VALUES ('s1', 2, 'Show', 'Show CN', NULL, NULL, NULL, NULL, '[]', '[]', 0, 'a')")
                db.execute("INSERT INTO entity VALUES ('ep1', 'episode', 'Ep', 'active')")
                db.execute("INSERT INTO episodes VALUES ('ep1', 's1', 1, 0, 'Ep', NULL, NULL, 'desc', NULL, NULL, 'a')")
                db.execute("INSERT INTO entity VALUES ('ep2', 'episode', 'Ep2', 'active')")
                db.execute("INSERT INTO episodes VALUES ('ep2', 's1', 2, 0, 'Ep2', NULL, NULL, '', NULL, NULL, 'a')")
                db.execute("INSERT INTO build VALUES ('b', 'now', NULL, 'test', 'done', '')")
                db.execute("INSERT INTO source_document VALUES ('d1', 'bgm', '1', 'Review', NULL, 'review body', '{}', 'h1', NULL, 'a')")
                db.execute("INSERT INTO source_document VALUES ('d2', 'bgm', '2', 'Other', NULL, 'other', '{}', 'h2', NULL, 'a')")
                db.execute("INSERT INTO community_document VALUES ('d1', 'bgm_review', 's1', NULL, '', NULL, NULL, 'unknown', 1, 0, 0)")
                db.execute("INSERT INTO community_document VALUES ('d2', 'other', 's1', NULL, '', NULL, NULL, 'unknown', 1, 0, 0)")
                db.execute("INSERT INTO document_chunk VALUES ('c1', 'd1', 0, 0, 12, 'chunk body', 'ch', 'a')")
                db.execute("INSERT INTO document_chunk VALUES ('c2', 'd2', 0, 0, 5, 'ignored', 'ch2', 'a')")
                db.execute("INSERT INTO community_report VALUES ('s1', 'Summary', '[\"good\"]', '[]', '[\"point\"]', '[\"ep1\"]', '[\"dispute\"]', '{}', 'd', 'p', 'p', 'm', 'r', 'b')")
                db.execute("INSERT INTO discussion_theme VALUES ('t1', 's1', 'ep1', 'Theme', 'aspect', ?, '{}', 'm', 'v', 'b')", (json.dumps([{'quote': 'first'}, {'quote': 'second'}, {'quote': 'third'}, {'quote': ''}]),))
                db.execute("INSERT INTO discussion_theme VALUES ('t2', 's1', 'ep1', '', 'aspect', '[]', '{}', 'm', 'v', 'b')")
                db.commit()

            sources = list(iter_vector_sources(database))
            self.assertEqual([item.source_type for item in sources], ["episode_card", "community_chunk", "community_report", "discussion_theme"])
            self.assertEqual(sources[0].schema_version, CARD_SCHEMA_VERSION)
            self.assertEqual(sources[0].episode_entity_id, "ep1")
            self.assertEqual(sources[1].document_id, "d1")
            self.assertIn("chunk body", sources[1].text)
            self.assertEqual(sources[1].metadata["source_kind"], "bgm_review")
            self.assertIn("Summary", sources[2].text)
            self.assertIn("dispute", sources[2].text)
            self.assertIn("first", sources[3].text)
            self.assertIn("second", sources[3].text)
            self.assertNotIn("third", sources[3].text)
            self.assertEqual([item.source_id for item in sources if item.source_type == "discussion_theme"], ["t1"])


if __name__ == "__main__":
    unittest.main()

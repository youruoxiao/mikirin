import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from mikirin.agent.tools import DatabaseQueryTool
from mikirin.layout import ProjectLayout
from mikirin.publish import Publisher


class DatabaseQueryToolTests(unittest.TestCase):
    def test_predefined_staff_and_episode_queries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = ProjectLayout(root)
            database = Publisher(layout).create_empty_build(
                build_id="build-test", code_version="test"
            )
            with closing(sqlite3.connect(database)) as db, db:
                db.execute(
                    """
                    INSERT INTO ingestion_run(
                        id, source, started_at, completed_at, source_version,
                        parser_version, status
                    ) VALUES ('seed', 'fixture', 'now', 'now', '1', '1', 'loaded')
                    """
                )
                db.execute(
                    """
                    INSERT INTO source_asset(
                        id, ingestion_run_id, local_path, content_hash, fetched_at
                    ) VALUES ('asset', 'seed', 'fixture', 'sha256:seed', 'now')
                    """
                )
                db.execute("INSERT INTO entity VALUES ('s', 'subject', '作品', 'active')")
                db.execute("INSERT INTO subjects(entity_id,type,name,source_asset_id) VALUES ('s',2,'作品','asset')")
                db.execute("INSERT INTO entity VALUES ('p', 'person', '监督', 'active')")
                db.execute("INSERT INTO persons(entity_id,name,source_asset_id) VALUES ('p','监督','asset')")
                db.execute("INSERT INTO rel_subject_person VALUES ('s','p','导演',NULL,'asset')")
                for episode_id, sort, episode_type in (("e1", 1, 0), ("e2", 2, 0), ("sp", 1, 1)):
                    db.execute(
                        "INSERT INTO entity VALUES (?, 'episode', ?, 'active')",
                        (episode_id, episode_id),
                    )
                    db.execute(
                        """
                        INSERT INTO episodes(
                            entity_id, subject_entity_id, sort, type, name, source_asset_id
                        ) VALUES (?, 's', ?, ?, ?, 'asset')
                        """,
                        (episode_id, sort, episode_type, episode_id),
                    )

            tool = DatabaseQueryTool(database)
            episodes = tool._query("episodes", "s", 20)
            self.assertEqual(
                [row["episode_entity_id"] for row in episodes["rows"]],
                ["e1", "e2", "sp"],
            )
            staff = tool._query("staff", "s", 20)
            self.assertEqual(staff["rows"][0]["name"], "监督")
            self.assertNotIn("sql", tool.parameters["properties"])


if __name__ == "__main__":
    unittest.main()

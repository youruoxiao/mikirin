import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from mikirin.layout import ProjectLayout
from mikirin.publish import Publisher
from mikirin.query import KnowledgeQueries
from mikirin.search_index import SearchIndexer
from mikirin.sources.supplementary_loader import SupplementaryStagingLoader, _text_chunks


class BgmWebLoaderTests(unittest.TestCase):
    @staticmethod
    def _create_seeded_database(root: Path, build_id: str) -> Path:
        database = Publisher(ProjectLayout(root)).create_empty_build(
            build_id=build_id, code_version="test"
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
                ) VALUES ('seed-asset', 'seed', 'fixture', 'sha256:seed', 'now')
                """
            )
            db.execute(
                "INSERT INTO entity VALUES ('bgm:subject:42', 'subject', 'Work', 'active')"
            )
            db.execute(
                """
                INSERT INTO subjects(entity_id, type, name, source_asset_id)
                VALUES ('bgm:subject:42', 2, 'Work', 'seed-asset')
                """
            )
            db.execute(
                "INSERT INTO entity VALUES ('bgm:episode:99', 'episode', 'Episode', 'active')"
            )
            db.execute(
                """
                INSERT INTO episodes(
                    entity_id, subject_entity_id, sort, type, name, source_asset_id
                ) VALUES ('bgm:episode:99', 'bgm:subject:42', 1, 0, 'Episode', 'seed-asset')
                """
            )
        return database

    def test_parallel_workers_match_sequential_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            staging.mkdir()
            (staging / "manifest.json").write_text(
                json.dumps({"source_version": "fixture"}), encoding="utf-8"
            )
            documents = [
                {
                    "document_id": "doc:subject",
                    "source_kind": "bgm_subject_comment",
                    "subject_entity_id": "bgm:subject:42",
                    "raw_content": "subject comment",
                    "content_hash": "sha256:subject",
                    "metadata_text": ["meta"],
                },
                {
                    "document_id": "doc:episode",
                    "source_kind": "bgm_ep_comment",
                    "source_record_id": "episode-record",
                    "subject_entity_id": "bgm:subject:42",
                    "episode_entity_id": "bgm:episode:99",
                    "raw_content": ("long episode comment " * 100),
                    "content_hash": "sha256:episode",
                    "rating": 8,
                    "redistributable": True,
                },
                {
                    "document_id": "doc:dangling-subject",
                    "source_kind": "bgm_review",
                    "subject_entity_id": "bgm:subject:404",
                    "raw_content": "missing subject",
                    "content_hash": "sha256:missing-subject",
                },
                {
                    "document_id": "doc:dangling-episode",
                    "source_kind": "bgm_ep_comment",
                    "subject_entity_id": "bgm:subject:42",
                    "episode_entity_id": "bgm:episode:404",
                    "raw_content": "missing episode",
                    "content_hash": "sha256:missing-episode",
                },
            ]
            (staging / "documents.jsonl").write_text(
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in documents),
                encoding="utf-8",
            )

            databases = [
                self._create_seeded_database(root / "sequential", "build-sequential"),
                self._create_seeded_database(root / "parallel", "build-parallel"),
            ]
            reports = [
                SupplementaryStagingLoader().load_bgm_web(
                    staging_directory=staging,
                    database_path=database,
                    source_version="fixture",
                    workers=workers,
                )
                for database, workers in zip(databases, (1, 2), strict=True)
            ]
            self.assertEqual(reports[0].counts, reports[1].counts)
            self.assertEqual(reports[0].counts["documents"], 2)
            self.assertEqual(reports[0].counts["dangling_subjects"], 1)
            self.assertEqual(reports[0].counts["dangling_episodes"], 1)

            snapshots = []
            for database in databases:
                with closing(sqlite3.connect(database)) as db:
                    snapshots.append(
                        {
                            table: db.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                            for table in (
                                "source_document",
                                "community_document",
                                "document_chunk",
                            )
                        }
                    )
            self.assertEqual(snapshots[0], snapshots[1])

    def test_workers_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "workers"):
            SupplementaryStagingLoader().load_bgm_web(
                staging_directory=Path("unused"),
                database_path=Path("unused.db"),
                source_version="fixture",
                workers=0,
            )

    def test_community_documents_keep_entity_anchor_and_provenance(self) -> None:
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
                    ) VALUES ('seed-asset', 'seed', 'fixture', 'sha256:seed', 'now')
                    """
                )
                db.execute("INSERT INTO entity VALUES ('bgm:subject:42', 'subject', '作品', 'active')")
                db.execute(
                    """
                    INSERT INTO subjects(
                        entity_id, type, name, source_asset_id
                    ) VALUES ('bgm:subject:42', 2, '作品', 'seed-asset')
                    """
                )
                db.execute(
                    """
                    INSERT INTO entity_alias(
                        id, entity_id, alias_text, normalized_alias, source_kind,
                        status, confidence, source_asset_id, build_id, license_id
                    ) VALUES (
                        'alias:work', 'bgm:subject:42', '作品', '作品', 'fixture',
                        'active', 1.0, 'seed-asset', 'build-test', 'fixture'
                    )
                    """
                )
                db.execute("INSERT INTO entity VALUES ('bgm:episode:99', 'episode', '第一话', 'active')")
                db.execute(
                    """
                    INSERT INTO episodes(
                        entity_id, subject_entity_id, sort, type, name, source_asset_id
                    ) VALUES ('bgm:episode:99', 'bgm:subject:42', 1, 0, '第一话', 'seed-asset')
                    """
                )

            staging = layout.staging / "bgm_web" / "fixture"
            staging.mkdir(parents=True)
            (staging / "manifest.json").write_text(
                json.dumps({"source_version": "fixture", "documents_hash": "sha256:fixture"}),
                encoding="utf-8",
            )
            document = {
                "document_id": "bgm:web-comment:abc",
                "source_kind": "bgm_ep_comment",
                "source_record_id": "123",
                "subject_entity_id": "bgm:subject:42",
                "episode_entity_id": "bgm:episode:99",
                "title": "",
                "url": "https://bangumi.tv/ep/99",
                "raw_content": "这一集的演出很有意思",
                "content_hash": "sha256:content",
                "fetched_at": "2026-07-27T00:00:00+00:00",
                "author_hash": "a" * 64,
                "rating": 9,
                "posted_at_raw": "@ 2026-7-27",
                "license_id": "bangumi-user-content",
                "redistributable": False,
                "attribution_required": True,
                "commercial_use_allowed": False,
                "raw_relative_path": "data/raw/example.html",
                "source_asset_hash": "sha256:raw",
            }
            (staging / "documents.jsonl").write_text(
                json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8"
            )

            report = SupplementaryStagingLoader().load_bgm_web(
                staging_directory=staging,
                database_path=database,
                source_version="fixture",
            )
            self.assertEqual(report.counts["documents"], 1)
            self.assertEqual(report.counts["chunks"], 1)
            self.assertEqual(report.counts["episode_comments"], 1)
            SearchIndexer.rebuild(database)
            context = KnowledgeQueries.search_anime_context(
                database, "我想聊作品的演出"
            )
            with closing(sqlite3.connect(database)) as db:
                row = db.execute(
                    """
                    SELECT c.subject_entity_id, c.episode_entity_id, c.rating,
                           c.redistributable, c.attribution_required, d.content
                    FROM community_document c
                    JOIN source_document d ON d.id=c.document_id
                    """
                ).fetchone()
                fts = db.execute(
                    "SELECT document_id FROM document_fts WHERE document_fts MATCH ?",
                    ('"有意思"',),
                ).fetchone()
                chunk_fts = db.execute(
                    "SELECT document_id FROM document_chunk_fts WHERE document_chunk_fts MATCH ?",
                    ('"有意思"',),
                ).fetchone()
            self.assertEqual(
                row,
                ("bgm:subject:42", "bgm:episode:99", 9, 0, 1, "这一集的演出很有意思"),
            )
            self.assertEqual(fts[0], "bgm:web-comment:abc")
            self.assertEqual(chunk_fts[0], "bgm:web-comment:abc")
            self.assertEqual(context["entities"][0]["entity_id"], "bgm:subject:42")
            self.assertEqual(context["documents"][0]["content"], "这一集的演出很有意思")

    def test_long_text_chunks_overlap_and_cover_tail(self) -> None:
        content = "甲" * 900 + "\n" + "乙" * 900 + "\n" + "丙" * 900
        chunks = list(_text_chunks(content))
        self.assertGreater(len(chunks), 2)
        self.assertEqual(chunks[0][1], 0)
        self.assertEqual(chunks[-1][2], len(content))
        self.assertLess(chunks[1][1], chunks[0][2])


if __name__ == "__main__":
    unittest.main()

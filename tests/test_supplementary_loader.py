import json
from contextlib import closing
import tempfile
import unittest
from pathlib import Path

from mikirin.layout import ProjectLayout
from mikirin.publish import Publisher
from mikirin.sources.supplementary_loader import SupplementaryStagingLoader
from mikirin.search_index import SearchIndexer
from mikirin.query import KnowledgeQueries


class SupplementaryLoaderTests(unittest.TestCase):
    def test_moegirl_candidates_remain_unresolved_and_atwiki_is_document_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = ProjectLayout(root)
            database = Publisher(layout).create_empty_build(
                build_id="build-test", code_version="test"
            )

            moegirl = root / "moegirl"
            moegirl.mkdir()
            (moegirl / "manifest.json").write_text("{}", encoding="utf-8")
            (moegirl / "characters.jsonl").write_text(
                json.dumps(
                    {
                        "entity_id": "uuid5:character:left",
                        "source_entity_id": "泉此方",
                        "raw_name": "泉此方",
                        "aliases": ["泉此方", "宅方"],
                        "attributes": ["宅女"],
                        "voice_actors": [],
                        "gender": "female",
                        "subjects": [],
                        "fields": {},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (moegirl / "mapping_candidates.jsonl").write_text(
                json.dumps(
                    {
                        "left_entity_id": "uuid5:character:left",
                        "right_entity_id": "bgm:character:123",
                        "confidence": 0.8,
                        "decision_source": "fixture",
                        "evidence": {"kind": "fixture"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            loader = SupplementaryStagingLoader()
            report = loader.load_moegirl(
                staging_directory=moegirl,
                database_path=database,
                source_version="fixture",
                build_id="build-test",
            )
            self.assertEqual(report.counts["mapping_candidates"], 1)

            atwiki = root / "atwiki"
            atwiki.mkdir()
            (atwiki / "manifest.json").write_text("{}", encoding="utf-8")
            (atwiki / "pages.jsonl").write_text(
                json.dumps(
                    {
                        "source_page_id": 106,
                        "title": "Ｘ 劇場版",
                        "url": "https://example.invalid/106",
                        "plain_text": "原画：誰か",
                        "role_sections": [{"role": "原画", "content": "誰か"}],
                        "page_kind": "unclassified",
                        "content_hash": "sha256:abc",
                        "fetched_at": "2026-07-26T00:00:00+00:00",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            report = loader.load_atwiki(
                staging_directory=atwiki,
                database_path=database,
                source_version="fixture",
            )
            self.assertEqual(report.counts["documents"], 1)
            self.assertEqual(report.counts["role_sections"], 1)
            search = SearchIndexer.rebuild(database)
            self.assertEqual(search.entities, 1)
            self.assertEqual(search.documents, 1)
            exact_hits = KnowledgeQueries.search_entities(database, "宅方")
            self.assertEqual(exact_hits[0].entity_id, "uuid5:character:left")

            import sqlite3

            with closing(sqlite3.connect(database)) as db:
                status = db.execute(
                    "SELECT status FROM entity WHERE id='uuid5:character:left'"
                ).fetchone()[0]
                candidate = db.execute("SELECT status FROM resolution_candidate").fetchone()[0]
                documents = db.execute("SELECT count(*) FROM source_document").fetchone()[0]
                role_sections = db.execute(
                    "SELECT count(*) FROM document_role_section"
                ).fetchone()[0]
                alias_hit = db.execute(
                    "SELECT entity_id FROM entity_fts WHERE entity_fts MATCH ?",
                    ('"泉此方"',),
                ).fetchone()[0]
            self.assertEqual(status, "provisional")
            self.assertEqual(candidate, "candidate")
            self.assertEqual(documents, 1)
            self.assertEqual(role_sections, 1)
            self.assertEqual(alias_hit, "uuid5:character:left")


if __name__ == "__main__":
    unittest.main()

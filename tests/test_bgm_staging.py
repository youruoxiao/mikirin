import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from mikirin.layout import ProjectLayout
from mikirin.sources.bgm_staging import BgmArchiveStager


class BgmStagingTests(unittest.TestCase):
    def test_staging_adds_stable_global_ids(self) -> None:
        records = {
            "subject.jsonlines": {"id": 1, "name": "Work", "type": 2},
            "person.jsonlines": {"id": 2, "name": "Person"},
            "character.jsonlines": {"id": 3, "name": "Character"},
            "episode.jsonlines": {"id": 4, "subject_id": 1, "sort": 1, "type": 0},
            "subject-relations.jsonlines": {
                "subject_id": 1,
                "related_subject_id": 1,
                "relation_type": 1,
            },
            "subject-characters.jsonlines": {
                "subject_id": 1,
                "character_id": 3,
                "type": 1,
            },
            "subject-persons.jsonlines": {
                "subject_id": 1,
                "person_id": 2,
                "position": "原画",
            },
            "person-characters.jsonlines": {
                "person_id": 2,
                "subject_id": 1,
                "character_id": 3,
            },
            "person-relations.jsonlines": {
                "person_type": "prsn",
                "person_id": 2,
                "related_person_id": 2,
                "relation_type": 1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name, record in records.items():
                    archive.writestr(name, json.dumps(record, ensure_ascii=False) + "\n")
            layout = ProjectLayout(root / "project")
            report = BgmArchiveStager(layout).stage(
                archive_path,
                source_version="fixture",
                source_asset_id="bgm_archive:sha256:test",
            )
            self.assertTrue(all(value == 1 for value in report.counts.values()))
            subject_path = layout.staging / "bgm_archive" / "fixture" / "subjects.jsonl"
            subject = json.loads(subject_path.read_text(encoding="utf-8"))
            self.assertEqual(subject["entity_id"], "bgm:subject:1")
            self.assertEqual(subject["source_asset_id"], "bgm_archive:sha256:test")


if __name__ == "__main__":
    unittest.main()


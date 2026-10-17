import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from mikirin.sources.bgm_archive import ARCHIVE_KINDS, ArchiveInventory, classify_member


class BgmArchiveTests(unittest.TestCase):
    def test_member_names_are_classified_across_common_spellings(self) -> None:
        self.assertEqual(classify_member("dump/subject.jsonlines"), "subjects")
        self.assertEqual(classify_member("subject_relations.jsonl"), "subject_relations")
        self.assertEqual(classify_member("person-characters.ndjson"), "person_characters")
        self.assertIsNone(classify_member("README.md"))

    def test_inventory_and_streaming_json_validation(self) -> None:
        names = {
            "subjects": "subject.jsonlines",
            "persons": "person.jsonlines",
            "characters": "character.jsonlines",
            "episodes": "episode.jsonlines",
            "subject_relations": "subject-relations.jsonlines",
            "subject_characters": "subject-characters.jsonlines",
            "subject_persons": "subject-persons.jsonlines",
            "person_characters": "person-characters.jsonlines",
            "person_relations": "person-relations.jsonlines",
        }
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for index, kind in enumerate(ARCHIVE_KINDS):
                    archive.writestr(names[kind], json.dumps({"id": index}) + "\n")
            inventory = ArchiveInventory.inspect(archive_path)
            self.assertEqual(inventory.missing_required, ())
            self.assertEqual(inventory.count_records(), {kind: 1 for kind in ARCHIVE_KINDS})

    def test_invalid_json_reports_member_and_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("subject.jsonlines", "{}\nnot-json\n")
            inventory = ArchiveInventory.inspect(archive_path)
            with self.assertRaisesRegex(ValueError, "第 2 行"):
                list(inventory.iter_records("subjects"))


if __name__ == "__main__":
    unittest.main()


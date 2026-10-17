import json
import tempfile
import unittest
from pathlib import Path

from mikirin.layout import ProjectLayout
from mikirin.sources.moegirl_staging import MoegirlStager


class MoegirlStagingTests(unittest.TestCase):
    def test_character_aliases_and_bgm_candidates_are_staged_without_auto_merge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = ProjectLayout(Path(directory))
            version = "fixture"
            raw = layout.raw / "moegirl_dataset" / version
            files = {
                "moegirl/extra_processed.json": {
                    "泉此方": {"本名": ["泉こなた"], "别名": ["宅方"], "萌点": ["宅女"]}
                },
                "moegirl/char_index.json": ["泉此方"],
                "moegirl/char2attr.json": {"泉此方": ["宅女"]},
                "moegirl/char2cv.json": {"泉此方": ["平野绫"]},
                "moegirl/char2gender.json": {"泉此方": "female"},
                "moegirl/char2subject.json": {"泉此方": ["幸运星"]},
                "bangumi/moegirl2bgm.json": {"泉此方": [123]},
            }
            for relative, value in files.items():
                path = raw / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            manifest = layout.manifests / "moegirl_dataset" / f"{version}.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {"expected_files": len(files), "files": {name: {} for name in files}}
                ),
                encoding="utf-8",
            )
            report = MoegirlStager(layout).stage(source_version=version)
            self.assertEqual(report.characters, 1)
            self.assertEqual(report.mapping_candidates, 1)
            character = json.loads(
                (layout.staging / "moegirl_dataset" / version / "characters.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("宅方", character["aliases"])
            self.assertTrue(character["entity_id"].startswith("uuid5:character:"))
            candidate = json.loads(
                (
                    layout.staging
                    / "moegirl_dataset"
                    / version
                    / "mapping_candidates.jsonl"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(candidate["decision"], "candidate")
            self.assertEqual(candidate["right_entity_id"], "bgm:character:123")


if __name__ == "__main__":
    unittest.main()


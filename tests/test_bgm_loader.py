import json
import sqlite3
import tempfile
import unittest
import zipfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from mikirin.assets import RawAsset
from mikirin.layout import ProjectLayout
from mikirin.publish import Publisher
from mikirin.sources.bgm_loader import BgmStagingLoader
from mikirin.sources.bgm_staging import BgmArchiveStager


class BgmLoaderTests(unittest.TestCase):
    def test_fixture_stages_and_loads_into_content_database(self) -> None:
        records = {
            "subject.jsonlines": {
                "id": 1,
                "name": "Work",
                "name_cn": "作品",
                "type": 2,
                "score": 8.1,
            },
            "person.jsonlines": {"id": 1, "name": "Person", "type": 1},
            "character.jsonlines": {"id": 1, "name": "Character", "role": 1},
            "episode.jsonlines": {"id": 1, "subject_id": 1, "sort": 1, "type": 0},
            "subject-relations.jsonlines": {
                "subject_id": 1,
                "related_subject_id": 1,
                "relation_type": 1,
            },
            "subject-characters.jsonlines": {
                "subject_id": 1,
                "character_id": 1,
                "type": 1,
            },
            "subject-persons.jsonlines": {
                "subject_id": 1,
                "person_id": 1,
                "position": "原画",
            },
            "person-characters.jsonlines": {
                "person_id": 1,
                "subject_id": 1,
                "character_id": 1,
            },
            "person-relations.jsonlines": {
                "person_type": "prsn",
                "person_id": 1,
                "related_person_id": 1,
                "relation_type": 1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = ProjectLayout(root / "project")
            archive_path = root / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for name, record in records.items():
                    archive.writestr(name, json.dumps(record, ensure_ascii=False) + "\n")
            asset = RawAsset(
                id="bgm_archive:sha256:test",
                source="bgm_archive",
                source_version="fixture",
                original_name="fixture.zip",
                relative_path="data/raw/bgm_archive/fixture.zip",
                content_hash="sha256:test",
                byte_size=archive_path.stat().st_size,
                ingested_at=datetime.now(UTC).isoformat(),
                license_id="test",
                redistributable=False,
                attribution_required=False,
                commercial_use_allowed=False,
            )
            BgmArchiveStager(layout).stage(
                archive_path,
                source_version="fixture",
                source_asset_id=asset.id,
            )
            publisher = Publisher(layout)
            database = publisher.create_empty_build(build_id="build-fixture", code_version="test")
            report = BgmStagingLoader().load(
                staging_directory=layout.staging / "bgm_archive" / "fixture",
                database_path=database,
                build_id="build-fixture",
                asset=asset,
            )
            self.assertEqual(report.counts["subjects"], 1)
            self.assertGreaterEqual(report.counts["aliases"], 4)
            Publisher.validate(database)
            with closing(sqlite3.connect(database)) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM entity").fetchone()[0], 4)
                self.assertEqual(db.execute("SELECT name_cn FROM subjects").fetchone()[0], "作品")

    def test_workers_one_and_two_produce_equivalent_results(self) -> None:
        records = {
            "subject.jsonlines": {
                "id": 1,
                "name": "Work",
                "name_cn": "作品",
                "type": 2,
                "score": 8.1,
            },
            "person.jsonlines": {"id": 1, "name": "Person", "type": 1},
            "character.jsonlines": {"id": 1, "name": "Character", "role": 1},
            "episode.jsonlines": {"id": 1, "subject_id": 1, "sort": 1, "type": 0},
            "subject-relations.jsonlines": {
                "subject_id": 1,
                "related_subject_id": 1,
                "relation_type": 1,
            },
            "subject-characters.jsonlines": {
                "subject_id": 1,
                "character_id": 1,
                "type": 1,
            },
            "subject-persons.jsonlines": {
                "subject_id": 1,
                "person_id": 1,
                "position": "原画",
            },
            "person-characters.jsonlines": {
                "person_id": 1,
                "subject_id": 1,
                "character_id": 1,
            },
            "person-relations.jsonlines": {
                "person_type": "prsn",
                "person_id": 1,
                "related_person_id": 1,
                "relation_type": 1,
            },
        }
        tables = (
            "entity",
            "source_entity",
            "subjects",
            "persons",
            "characters",
            "episodes",
            "entity_alias",
            "rel_subject_subject",
            "rel_subject_character",
            "rel_subject_person",
            "rel_person_character",
            "rel_person_person",
        )
        snapshots: list[dict[str, list[tuple[object, ...]]]] = []
        reports = []
        for workers in (1, 2):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                layout = ProjectLayout(root / "project")
                archive_path = root / "fixture.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    for name, record in records.items():
                        if name == "subject.jsonlines":
                            archive.writestr(
                                name,
                                "".join(
                                    json.dumps(
                                        {
                                            **record,
                                            "id": subject_id,
                                            "name": f"Work {subject_id}",
                                        },
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                    for subject_id in range(1, 5_002)
                                ),
                            )
                        else:
                            archive.writestr(name, json.dumps(record, ensure_ascii=False) + "\n")
                asset = RawAsset(
                    id="bgm_archive:sha256:test",
                    source="bgm_archive",
                    source_version="fixture",
                    original_name="fixture.zip",
                    relative_path="data/raw/bgm_archive/fixture.zip",
                    content_hash="sha256:test",
                    byte_size=archive_path.stat().st_size,
                    ingested_at="2026-01-01T00:00:00+00:00",
                    license_id="test",
                    redistributable=False,
                    attribution_required=False,
                    commercial_use_allowed=False,
                )
                staging_report = BgmArchiveStager(layout).stage(
                    archive_path,
                    source_version="fixture",
                    source_asset_id=asset.id,
                )
                staging = layout.root / staging_report.output_directory
                database = Publisher(layout).create_empty_build(
                    build_id="build-fixture", code_version="test"
                )
                reports.append(
                    BgmStagingLoader().load(
                        staging_directory=staging,
                        database_path=database,
                        build_id="build-fixture",
                        asset=asset,
                        workers=workers,
                    )
                )
                with closing(sqlite3.connect(database)) as db:
                    snapshots.append(
                        {
                            table: db.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall()
                            for table in tables
                        }
                    )

        self.assertEqual(reports[0].counts, reports[1].counts)
        self.assertEqual(snapshots[0], snapshots[1])

    def test_non_positive_workers_is_rejected_before_database_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "workers"):
            BgmStagingLoader().load(
                staging_directory=Path("missing"),
                database_path=Path("missing.db"),
                build_id="missing",
                asset=None,  # type: ignore[arg-type]
                workers=0,
            )


if __name__ == "__main__":
    unittest.main()

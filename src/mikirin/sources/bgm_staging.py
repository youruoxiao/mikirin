from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mikirin.atomic import atomic_write_json
from mikirin.ids import authoritative_entity_id
from mikirin.layout import ProjectLayout
from mikirin.sources.bgm_archive import ARCHIVE_KINDS, ArchiveInventory


class RecordError(ValueError):
    pass


def _required(record: dict[str, Any], key: str) -> Any:
    value = record.get(key)
    if value is None or value == "":
        raise RecordError(f"缺少必填字段 {key!r}")
    return value


def _base(record: dict[str, Any], asset_id: str) -> dict[str, Any]:
    return {**record, "source_asset_id": asset_id}


def _subject(record: dict[str, Any], asset_id: str) -> dict[str, Any]:
    source_id = _required(record, "id")
    return {
        **_base(record, asset_id),
        "source_entity_id": str(source_id),
        "entity_id": authoritative_entity_id("bgm", "subject", source_id),
    }


def _person(record: dict[str, Any], asset_id: str) -> dict[str, Any]:
    source_id = _required(record, "id")
    return {
        **_base(record, asset_id),
        "source_entity_id": str(source_id),
        "entity_id": authoritative_entity_id("bgm", "person", source_id),
    }


def _character(record: dict[str, Any], asset_id: str) -> dict[str, Any]:
    source_id = _required(record, "id")
    return {
        **_base(record, asset_id),
        "source_entity_id": str(source_id),
        "entity_id": authoritative_entity_id("bgm", "character", source_id),
    }


def _episode(record: dict[str, Any], asset_id: str) -> dict[str, Any]:
    source_id = _required(record, "id")
    subject_id = _required(record, "subject_id")
    return {
        **_base(record, asset_id),
        "source_entity_id": str(source_id),
        "entity_id": authoritative_entity_id("bgm", "episode", source_id),
        "subject_entity_id": authoritative_entity_id("bgm", "subject", subject_id),
    }


def _subject_relation(record: dict[str, Any], asset_id: str) -> dict[str, Any]:
    return {
        **_base(record, asset_id),
        "subject_entity_id": authoritative_entity_id(
            "bgm", "subject", _required(record, "subject_id")
        ),
        "related_subject_entity_id": authoritative_entity_id(
            "bgm", "subject", _required(record, "related_subject_id")
        ),
    }


def _subject_character(record: dict[str, Any], asset_id: str) -> dict[str, Any]:
    return {
        **_base(record, asset_id),
        "subject_entity_id": authoritative_entity_id(
            "bgm", "subject", _required(record, "subject_id")
        ),
        "character_entity_id": authoritative_entity_id(
            "bgm", "character", _required(record, "character_id")
        ),
    }


def _subject_person(record: dict[str, Any], asset_id: str) -> dict[str, Any]:
    return {
        **_base(record, asset_id),
        "subject_entity_id": authoritative_entity_id(
            "bgm", "subject", _required(record, "subject_id")
        ),
        "person_entity_id": authoritative_entity_id(
            "bgm", "person", _required(record, "person_id")
        ),
    }


def _person_character(record: dict[str, Any], asset_id: str) -> dict[str, Any]:
    return {
        **_base(record, asset_id),
        "person_entity_id": authoritative_entity_id(
            "bgm", "person", _required(record, "person_id")
        ),
        "subject_entity_id": authoritative_entity_id(
            "bgm", "subject", _required(record, "subject_id")
        ),
        "character_entity_id": authoritative_entity_id(
            "bgm", "character", _required(record, "character_id")
        ),
    }


def _person_relation(record: dict[str, Any], asset_id: str) -> dict[str, Any]:
    person_type = _required(record, "person_type")
    entity_type = {"prsn": "person", "crt": "character"}.get(str(person_type))
    if entity_type is None:
        raise RecordError(f"未知 person_type: {person_type!r}")
    return {
        **_base(record, asset_id),
        "person_entity_id": authoritative_entity_id(
            "bgm", entity_type, _required(record, "person_id")
        ),
        "related_person_entity_id": authoritative_entity_id(
            "bgm", entity_type, _required(record, "related_person_id")
        ),
    }


_NORMALIZERS: dict[str, Callable[[dict[str, Any], str], dict[str, Any]]] = {
    "subjects": _subject,
    "persons": _person,
    "characters": _character,
    "episodes": _episode,
    "subject_relations": _subject_relation,
    "subject_characters": _subject_character,
    "subject_persons": _subject_person,
    "person_characters": _person_character,
    "person_relations": _person_relation,
}


@dataclass(frozen=True, slots=True)
class StagingReport:
    source_version: str
    source_asset_id: str
    counts: dict[str, int]
    rejected: dict[str, int]
    output_directory: str


class BgmArchiveStager:
    def __init__(self, layout: ProjectLayout) -> None:
        self.layout = layout

    def stage(
        self,
        archive_path: Path,
        *,
        source_version: str,
        source_asset_id: str,
    ) -> StagingReport:
        inventory = ArchiveInventory.inspect(archive_path)
        if inventory.missing_required:
            raise ValueError(f"BGM Archive 缺少必需成员: {inventory.missing_required}")

        target = self.layout.staging / "bgm_archive" / source_version
        if target.exists():
            raise FileExistsError(f"staging 版本已存在，拒绝覆盖: {target}")
        temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
        rejected_root = self.layout.rejected / "bgm_archive" / source_version
        temporary.mkdir(parents=True, exist_ok=False)
        rejected_root.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        rejected: dict[str, int] = {}
        try:
            for kind in ARCHIVE_KINDS:
                accepted_count = 0
                rejected_count = 0
                output_path = temporary / f"{kind}.jsonl"
                rejected_path = rejected_root / f"{kind}.jsonl"
                with output_path.open("w", encoding="utf-8", newline="\n") as output, rejected_path.open(
                    "w", encoding="utf-8", newline="\n"
                ) as rejected_output:
                    for line_number, record in enumerate(inventory.iter_records(kind), start=1):
                        try:
                            normalized = _NORMALIZERS[kind](record, source_asset_id)
                        except (RecordError, ValueError) as error:
                            rejected_output.write(
                                json.dumps(
                                    {
                                        "line_number": line_number,
                                        "error": str(error),
                                        "record": record,
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                            rejected_count += 1
                            continue
                        output.write(
                            json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                        accepted_count += 1
                counts[kind] = accepted_count
                rejected[kind] = rejected_count
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        report = StagingReport(
            source_version=source_version,
            source_asset_id=source_asset_id,
            counts=counts,
            rejected=rejected,
            output_directory=target.relative_to(self.layout.root).as_posix(),
        )
        atomic_write_json(target / "manifest.json", asdict(report))
        return report

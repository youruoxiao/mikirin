from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mikirin.atomic import atomic_write_json
from mikirin.ids import authoritative_entity_id, provisional_entity_id
from mikirin.layout import ProjectLayout
from mikirin.sources.moegirl import MOEGIRL_COMMIT


def _load_json(root: Path, relative_path: str) -> Any:
    with (root / relative_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    return []


@dataclass(frozen=True, slots=True)
class MoegirlStagingReport:
    source_version: str
    characters: int
    mapping_candidates: int
    output_directory: str


class MoegirlStager:
    def __init__(self, layout: ProjectLayout) -> None:
        self.layout = layout

    def stage(self, *, source_version: str = MOEGIRL_COMMIT) -> MoegirlStagingReport:
        raw_root = self.layout.raw / "moegirl_dataset" / source_version
        manifest_path = self.layout.manifests / "moegirl_dataset" / f"{source_version}.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if len(manifest.get("files", {})) != manifest.get("expected_files"):
            raise ValueError("萌娘数据文件尚未全部下载并校验")

        extra = _load_json(raw_root, "moegirl/extra_processed.json")
        char_index = _load_json(raw_root, "moegirl/char_index.json")
        char2attr = _load_json(raw_root, "moegirl/char2attr.json")
        char2cv = _load_json(raw_root, "moegirl/char2cv.json")
        char2gender = _load_json(raw_root, "moegirl/char2gender.json")
        char2subject = _load_json(raw_root, "moegirl/char2subject.json")
        moegirl2bgm = _load_json(raw_root, "bangumi/moegirl2bgm.json")
        titles = sorted(
            set(char_index)
            | set(extra)
            | set(char2attr)
            | set(char2cv)
            | set(char2gender)
            | set(char2subject)
        )

        target = self.layout.staging / "moegirl_dataset" / source_version
        if target.exists():
            raise FileExistsError(f"萌娘 staging 已存在，拒绝覆盖: {target}")
        temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir(parents=True, exist_ok=False)
        character_count = 0
        candidate_count = 0
        try:
            with (temporary / "characters.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as characters_output, (temporary / "mapping_candidates.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as candidates_output:
                for title in titles:
                    fields = extra.get(title) if isinstance(extra.get(title), dict) else {}
                    aliases = []
                    for alias in [title, *_string_list(fields.get("本名")), *_string_list(fields.get("别名"))]:
                        if alias not in aliases:
                            aliases.append(alias)
                    entity_id = provisional_entity_id(
                        source="moegirl_dataset",
                        entity_type="character",
                        source_record_id=title,
                        record_locator=f"{source_version}:moegirl-title",
                        raw_name=title,
                    )
                    character = {
                        "source": "moegirl_dataset",
                        "source_version": source_version,
                        "source_entity_id": title,
                        "entity_id": entity_id,
                        "raw_name": title,
                        "aliases": aliases,
                        "attributes": char2attr.get(title, _string_list(fields.get("萌点"))),
                        "voice_actors": char2cv.get(title, _string_list(fields.get("声优"))),
                        "gender": char2gender.get(title),
                        "subjects": char2subject.get(title, []),
                        "fields": fields,
                    }
                    characters_output.write(
                        json.dumps(character, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    character_count += 1
                    for bgm_id in moegirl2bgm.get(title, []):
                        candidate = {
                            "left_entity_id": entity_id,
                            "left_source": "moegirl_dataset",
                            "left_source_entity_id": title,
                            "right_entity_id": authoritative_entity_id(
                                "bgm", "character", bgm_id
                            ),
                            "right_source": "bgm",
                            "right_source_entity_id": str(bgm_id),
                            "decision": "candidate",
                            "decision_source": "moegirl_dataset_mapping",
                            "confidence": 0.8,
                            "evidence": {
                                "dataset_commit": source_version,
                                "mapping_file": "bangumi/moegirl2bgm.json",
                            },
                        }
                        candidates_output.write(
                            json.dumps(candidate, ensure_ascii=False, sort_keys=True) + "\n"
                        )
                        candidate_count += 1
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        report = MoegirlStagingReport(
            source_version=source_version,
            characters=character_count,
            mapping_candidates=candidate_count,
            output_directory=target.relative_to(self.layout.root).as_posix(),
        )
        atomic_write_json(target / "manifest.json", asdict(report))
        return report


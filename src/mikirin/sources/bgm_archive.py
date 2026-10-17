from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ARCHIVE_KINDS = (
    "subjects",
    "persons",
    "characters",
    "episodes",
    "subject_relations",
    "subject_characters",
    "subject_persons",
    "person_characters",
    "person_relations",
)

_ALIASES = {
    "subject": "subjects",
    "subjects": "subjects",
    "person": "persons",
    "persons": "persons",
    "character": "characters",
    "characters": "characters",
    "episode": "episodes",
    "episodes": "episodes",
    "subject-relation": "subject_relations",
    "subject-relations": "subject_relations",
    "subject-character": "subject_characters",
    "subject-characters": "subject_characters",
    "subject-person": "subject_persons",
    "subject-persons": "subject_persons",
    "person-character": "person_characters",
    "person-characters": "person_characters",
    "person-relation": "person_relations",
    "person-relations": "person_relations",
}

_JSON_LINE_SUFFIX = re.compile(r"\.(?:jsonlines|jsonl|ndjson|json)(?:\.txt)?$", re.IGNORECASE)


def classify_member(member_name: str) -> str | None:
    basename = Path(member_name.replace("\\", "/")).name.lower()
    stem = _JSON_LINE_SUFFIX.sub("", basename).replace("_", "-")
    return _ALIASES.get(stem)


@dataclass(frozen=True, slots=True)
class ArchiveInventory:
    archive_path: Path
    members: dict[str, str]

    @classmethod
    def inspect(cls, archive_path: Path) -> "ArchiveInventory":
        archive_path = archive_path.resolve()
        members: dict[str, str] = {}
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                kind = classify_member(info.filename)
                if kind is None:
                    continue
                if kind in members:
                    raise ValueError(
                        f"压缩包中同一数据类型出现多个成员: {kind}: "
                        f"{members[kind]!r}, {info.filename!r}"
                    )
                members[kind] = info.filename
        return cls(archive_path=archive_path, members=members)

    @property
    def missing_required(self) -> tuple[str, ...]:
        return tuple(kind for kind in ARCHIVE_KINDS if kind not in self.members)

    def iter_records(self, kind: str) -> Iterator[dict[str, Any]]:
        if kind not in self.members:
            raise KeyError(f"压缩包不包含 {kind}")
        with zipfile.ZipFile(self.archive_path) as archive:
            with archive.open(self.members[kind], "r") as raw:
                for line_number, encoded_line in enumerate(raw, start=1):
                    if not encoded_line.strip():
                        continue
                    try:
                        value = json.loads(encoded_line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ValueError(
                            f"{self.members[kind]} 第 {line_number} 行不是合法 UTF-8 JSON"
                        ) from error
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"{self.members[kind]} 第 {line_number} 行必须是 JSON object"
                        )
                    yield value

    def count_records(self) -> dict[str, int]:
        return {kind: sum(1 for _ in self.iter_records(kind)) for kind in self.members}


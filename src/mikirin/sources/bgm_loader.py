from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from mikirin.assets import RawAsset
from mikirin.sources.bgm_aliases import normalize_alias, record_aliases


T = TypeVar("T")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any, default: Any) -> str:
    if value is None:
        value = default
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON object")
            yield value


def _batches(values: Iterable[T], size: int = 5_000) -> Iterator[list[T]]:
    batch: list[T] = []
    for value in values:
        batch.append(value)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _parse_jsonl_batch(
    path_text: str, lines: list[tuple[int, str]]
) -> list[dict[str, Any]]:
    """Parse one JSONL batch in a spawn-safe worker process."""
    values: list[dict[str, Any]] = []
    for line_number, line in lines:
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path_text} line {line_number} is not a JSON object")
        values.append(value)
    return values


def _parallel_batches(
    path: Path, workers: int, size: int = 5_000
) -> Iterator[list[dict[str, Any]]]:
    """Parse bounded JSONL batches in parallel and yield them in file order."""
    max_pending = max(1, workers * 2)
    pending: list[Future[list[dict[str, Any]]]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        with path.open("r", encoding="utf-8") as handle:
            lines: list[tuple[int, str]] = []
            for line_number, line in enumerate(handle, start=1):
                lines.append((line_number, line))
                if len(lines) < size:
                    continue
                pending.append(executor.submit(_parse_jsonl_batch, str(path), lines))
                lines = []
                if len(pending) >= max_pending:
                    yield pending.pop(0).result()
            if lines:
                pending.append(executor.submit(_parse_jsonl_batch, str(path), lines))
        while pending:
            yield pending.pop(0).result()


def _record_batches(path: Path, workers: int) -> Iterator[list[dict[str, Any]]]:
    if workers == 1:
        yield from _batches(_rows(path))
        return
    yield from _parallel_batches(path, workers)


def _source_url(kind: str, source_id: str) -> str:
    route = {"subject": "subject", "person": "person", "character": "character", "episode": "ep"}
    return f"https://bgm.tv/{route[kind]}/{source_id}"


@dataclass(frozen=True, slots=True)
class LoadReport:
    build_id: str
    ingestion_run_id: str
    counts: dict[str, int]


class BgmStagingLoader:
    def load(
        self,
        *,
        staging_directory: Path,
        database_path: Path,
        build_id: str,
        asset: RawAsset,
        parser_version: str = "bgm-loader-v1",
        workers: int = 1,
    ) -> LoadReport:
        if workers <= 0:
            raise ValueError("workers must be greater than zero")
        staging_directory = staging_directory.resolve()
        database_path = database_path.resolve()
        ingestion_run_id = f"ingest:bgm_archive:{asset.source_version}:{asset.content_hash}"
        started_at = _utc_now()
        counts: dict[str, int] = {}

        with closing(sqlite3.connect(database_path)) as db, db:
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("PRAGMA synchronous = OFF")
            if db.execute("SELECT 1 FROM build WHERE id = ?", (build_id,)).fetchone() is None:
                raise KeyError(f"构建记录不存在: {build_id}")
            db.execute(
                """
                INSERT INTO ingestion_run(
                    id, source, started_at, source_version, parser_version, status
                ) VALUES (?, 'bgm_archive', ?, ?, ?, 'loading')
                """,
                (ingestion_run_id, started_at, asset.source_version, parser_version),
            )
            db.execute(
                """
                INSERT INTO source_asset(
                    id, ingestion_run_id, local_path, content_hash, fetched_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    asset.id,
                    ingestion_run_id,
                    asset.relative_path,
                    asset.content_hash,
                    asset.ingested_at,
                ),
            )

            counts["subjects"], subject_aliases = self._load_subjects(
                db, staging_directory / "subjects.jsonl", asset.id, build_id, workers
            )
            counts["persons"], person_aliases = self._load_people(
                db, staging_directory / "persons.jsonl", asset.id, build_id, workers
            )
            counts["characters"], character_aliases = self._load_characters(
                db, staging_directory / "characters.jsonl", asset.id, build_id, workers
            )
            counts["aliases"] = subject_aliases + person_aliases + character_aliases
            counts["episodes"], counts["episodes_dangling"] = self._load_episodes(
                db, staging_directory / "episodes.jsonl", asset.id, workers
            )
            relation_loaders = {
                "subject_relations": self._load_subject_relations,
                "subject_characters": self._load_subject_characters,
                "subject_persons": self._load_subject_persons,
                "person_characters": self._load_person_characters,
                "person_relations": self._load_person_relations,
            }
            for kind, loader in relation_loaders.items():
                (
                    counts[kind],
                    counts[f"{kind}_dangling"],
                    counts[f"{kind}_duplicate"],
                ) = loader(
                    db, staging_directory / f"{kind}.jsonl", asset.id, workers
                )

            anomaly_suffixes = ("_dangling", "_duplicate")
            errors = sum(
                value for key, value in counts.items() if key.endswith(anomaly_suffixes)
            )
            total = sum(
                value for key, value in counts.items() if not key.endswith(anomaly_suffixes)
            )
            completed_at = _utc_now()
            db.execute(
                """
                UPDATE ingestion_run
                SET completed_at = ?, status = 'loaded', row_count = ?, error_count = ?
                WHERE id = ?
                """,
                (completed_at, total, errors, ingestion_run_id),
            )
            db.execute(
                "UPDATE build SET completed_at = ?, status = 'loaded' WHERE id = ?",
                (completed_at, build_id),
            )
        return LoadReport(build_id=build_id, ingestion_run_id=ingestion_run_id, counts=counts)

    @staticmethod
    def _entity_rows(batch: list[dict[str, Any]], entity_type: str) -> list[tuple[Any, ...]]:
        values = []
        for record in batch:
            name = record.get("name_cn") or record.get("name") or f"({entity_type}:{record['id']})"
            values.append((record["entity_id"], entity_type, str(name), "active"))
        return values

    @staticmethod
    def _source_entity_rows(
        batch: list[dict[str, Any]], kind: str, asset_id: str
    ) -> list[tuple[Any, ...]]:
        return [
            (
                record["entity_id"],
                "bgm",
                kind,
                record["source_entity_id"],
                record["entity_id"],
                str(record.get("name") or record.get("name_cn") or record["source_entity_id"]),
                _source_url(kind, record["source_entity_id"]),
                "resolved",
                1.0,
                asset_id,
            )
            for record in batch
        ]

    def _load_subjects(
        self,
        db: sqlite3.Connection,
        path: Path,
        asset_id: str,
        build_id: str,
        workers: int,
    ) -> tuple[int, int]:
        count = 0
        alias_count = 0
        for batch in _record_batches(path, workers):
            db.executemany("INSERT INTO entity VALUES (?, ?, ?, ?)", self._entity_rows(batch, "subject"))
            db.executemany(
                "INSERT INTO source_entity VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._source_entity_rows(batch, "subject", asset_id),
            )
            db.executemany(
                """
                INSERT INTO subjects(
                    entity_id, type, name, name_cn, infobox_raw, summary, date, platform,
                    tags_json, meta_tags_json, nsfw, source_asset_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["entity_id"],
                        int(r["type"]),
                        str(r.get("name") or ""),
                        r.get("name_cn"),
                        r.get("infobox"),
                        r.get("summary"),
                        r.get("date"),
                        str(r["platform"]) if r.get("platform") is not None else None,
                        _json(r.get("tags"), []),
                        _json(r.get("meta_tags"), []),
                        int(bool(r.get("nsfw", False))),
                        asset_id,
                    )
                    for r in batch
                ],
            )
            db.executemany(
                """
                INSERT INTO score_snapshot(
                    subject_entity_id, build_id, score, rank,
                    score_details_json, favorite_counts_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["entity_id"],
                        build_id,
                        r.get("score"),
                        r.get("rank"),
                        _json(r.get("score_details"), {}),
                        _json(r.get("favorite"), {}),
                    )
                    for r in batch
                ],
            )
            alias_count += self._load_aliases(db, batch, asset_id, build_id)
            count += len(batch)
        return count, alias_count

    def _load_people(
        self,
        db: sqlite3.Connection,
        path: Path,
        asset_id: str,
        build_id: str,
        workers: int,
    ) -> tuple[int, int]:
        count = 0
        alias_count = 0
        for batch in _record_batches(path, workers):
            db.executemany("INSERT INTO entity VALUES (?, ?, ?, ?)", self._entity_rows(batch, "person"))
            db.executemany(
                "INSERT INTO source_entity VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._source_entity_rows(batch, "person", asset_id),
            )
            db.executemany(
                """
                INSERT INTO persons(
                    entity_id, name, type, career_json, infobox_raw, summary,
                    comments, collects, source_asset_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["entity_id"],
                        str(r.get("name") or ""),
                        r.get("type"),
                        _json(r.get("career"), []),
                        r.get("infobox"),
                        r.get("summary"),
                        r.get("comments"),
                        r.get("collects"),
                        asset_id,
                    )
                    for r in batch
                ],
            )
            alias_count += self._load_aliases(db, batch, asset_id, build_id)
            count += len(batch)
        return count, alias_count

    def _load_characters(
        self,
        db: sqlite3.Connection,
        path: Path,
        asset_id: str,
        build_id: str,
        workers: int,
    ) -> tuple[int, int]:
        count = 0
        alias_count = 0
        for batch in _record_batches(path, workers):
            db.executemany(
                "INSERT INTO entity VALUES (?, ?, ?, ?)", self._entity_rows(batch, "character")
            )
            db.executemany(
                "INSERT INTO source_entity VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._source_entity_rows(batch, "character", asset_id),
            )
            db.executemany(
                """
                INSERT INTO characters(
                    entity_id, name, role, infobox_raw, summary,
                    comments, collects, source_asset_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["entity_id"],
                        str(r.get("name") or ""),
                        r.get("role"),
                        r.get("infobox"),
                        r.get("summary"),
                        r.get("comments"),
                        r.get("collects"),
                        asset_id,
                    )
                    for r in batch
                ],
            )
            alias_count += self._load_aliases(db, batch, asset_id, build_id)
            count += len(batch)
        return count, alias_count

    @staticmethod
    def _load_aliases(
        db: sqlite3.Connection,
        batch: list[dict[str, Any]],
        asset_id: str,
        build_id: str,
    ) -> int:
        values: list[tuple[Any, ...]] = []
        for record in batch:
            for alias in record_aliases(record):
                normalized = normalize_alias(alias)
                alias_id = "alias:" + uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{record['entity_id']}\0{normalized}\0bgm_archive",
                ).hex
                values.append(
                    (
                        alias_id,
                        record["entity_id"],
                        alias,
                        normalized,
                        "bgm_archive",
                        asset_id,
                        build_id,
                    )
                )
        before = db.total_changes
        db.executemany(
            """
            INSERT OR IGNORE INTO entity_alias(
                id, entity_id, alias_text, normalized_alias, lang,
                source_kind, status, confidence, source_asset_id,
                build_id, license_id
            ) VALUES (?, ?, ?, ?, NULL, ?, 'active', 1.0, ?, ?, 'upstream')
            """,
            values,
        )
        return db.total_changes - before

    @staticmethod
    def _filter_existing(
        db: sqlite3.Connection,
        batch: list[dict[str, Any]],
        requirements: tuple[tuple[str, str, str], ...],
    ) -> tuple[list[dict[str, Any]], int]:
        allowed: dict[str, set[str]] = {}
        for record_key, table, column in requirements:
            requested = {str(record[record_key]) for record in batch}
            if not requested:
                allowed[record_key] = set()
                continue
            placeholders = ",".join("?" for _ in requested)
            rows = db.execute(
                f"SELECT {column} FROM {table} WHERE {column} IN ({placeholders})",
                tuple(requested),
            )
            allowed[record_key] = {str(row[0]) for row in rows}
        valid = [
            record
            for record in batch
            if all(str(record[key]) in allowed[key] for key, _, _ in requirements)
        ]
        return valid, len(batch) - len(valid)

    def _load_episodes(
        self, db: sqlite3.Connection, path: Path, asset_id: str, workers: int
    ) -> tuple[int, int]:
        count = 0
        dangling = 0
        for batch in _record_batches(path, workers):
            batch, skipped = self._filter_existing(
                db, batch, (("subject_entity_id", "subjects", "entity_id"),)
            )
            dangling += skipped
            if not batch:
                continue
            db.executemany("INSERT INTO entity VALUES (?, ?, ?, ?)", self._entity_rows(batch, "episode"))
            db.executemany(
                "INSERT INTO source_entity VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._source_entity_rows(batch, "episode", asset_id),
            )
            db.executemany(
                """
                INSERT INTO episodes(
                    entity_id, subject_entity_id, sort, type, name, name_cn, airdate,
                    description, disc, duration, source_asset_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["entity_id"],
                        r["subject_entity_id"],
                        r.get("sort"),
                        r.get("type"),
                        r.get("name"),
                        r.get("name_cn"),
                        r.get("airdate"),
                        r.get("description"),
                        r.get("disc"),
                        str(r["duration"]) if r.get("duration") is not None else None,
                        asset_id,
                    )
                    for r in batch
                ],
            )
            count += len(batch)
        return count, dangling

    @staticmethod
    def _load_relation(
        db: sqlite3.Connection,
        path: Path,
        sql: str,
        transform: Any,
        requirements: tuple[tuple[str, str, str], ...],
        workers: int,
    ) -> tuple[int, int, int]:
        count = 0
        dangling = 0
        duplicates = 0
        for batch in _record_batches(path, workers):
            batch, skipped = BgmStagingLoader._filter_existing(db, batch, requirements)
            dangling += skipped
            before = db.total_changes
            insert_sql = sql.replace("INSERT INTO", "INSERT OR IGNORE INTO", 1)
            db.executemany(insert_sql, [transform(record) for record in batch])
            inserted = db.total_changes - before
            count += inserted
            duplicates += len(batch) - inserted
        return count, dangling, duplicates

    def _load_subject_relations(
        self, db: sqlite3.Connection, path: Path, asset_id: str, workers: int
    ) -> tuple[int, int, int]:
        return self._load_relation(
            db,
            path,
            "INSERT INTO rel_subject_subject VALUES (?, ?, ?, ?, ?)",
            lambda r: (
                r["subject_entity_id"],
                r["relation_type"],
                r["related_subject_entity_id"],
                r.get("order"),
                asset_id,
            ),
            (
                ("subject_entity_id", "subjects", "entity_id"),
                ("related_subject_entity_id", "subjects", "entity_id"),
            ),
            workers,
        )

    def _load_subject_characters(
        self, db: sqlite3.Connection, path: Path, asset_id: str, workers: int
    ) -> tuple[int, int, int]:
        return self._load_relation(
            db,
            path,
            "INSERT INTO rel_subject_character VALUES (?, ?, ?, ?, ?)",
            lambda r: (
                r["subject_entity_id"],
                r["character_entity_id"],
                r["type"],
                r.get("order"),
                asset_id,
            ),
            (
                ("subject_entity_id", "subjects", "entity_id"),
                ("character_entity_id", "characters", "entity_id"),
            ),
            workers,
        )

    def _load_subject_persons(
        self, db: sqlite3.Connection, path: Path, asset_id: str, workers: int
    ) -> tuple[int, int, int]:
        return self._load_relation(
            db,
            path,
            "INSERT INTO rel_subject_person VALUES (?, ?, ?, ?, ?)",
            lambda r: (
                r["subject_entity_id"],
                r["person_entity_id"],
                str(r["position"]),
                _json(r.get("appear_eps"), None),
                asset_id,
            ),
            (
                ("subject_entity_id", "subjects", "entity_id"),
                ("person_entity_id", "persons", "entity_id"),
            ),
            workers,
        )

    def _load_person_characters(
        self, db: sqlite3.Connection, path: Path, asset_id: str, workers: int
    ) -> tuple[int, int, int]:
        return self._load_relation(
            db,
            path,
            "INSERT INTO rel_person_character VALUES (?, ?, ?, ?, ?)",
            lambda r: (
                r["person_entity_id"],
                r["subject_entity_id"],
                r["character_entity_id"],
                r.get("summary"),
                asset_id,
            ),
            (
                ("person_entity_id", "persons", "entity_id"),
                ("subject_entity_id", "subjects", "entity_id"),
                ("character_entity_id", "characters", "entity_id"),
            ),
            workers,
        )

    def _load_person_relations(
        self, db: sqlite3.Connection, path: Path, asset_id: str, workers: int
    ) -> tuple[int, int, int]:
        return self._load_relation(
            db,
            path,
            "INSERT INTO rel_person_person VALUES (?, ?, ?, ?, ?, ?, ?)",
            lambda r: (
                str(r["person_type"]),
                r["person_entity_id"],
                r["related_person_entity_id"],
                r["relation_type"],
                int(bool(r.get("spoiler", False))),
                int(bool(r.get("ended", False))),
                asset_id,
            ),
            (
                ("person_entity_id", "entity", "id"),
                ("related_person_entity_id", "entity", "id"),
            ),
            workers,
        )

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatabaseQualityReport:
    database: str
    table_counts: dict[str, int]
    empty_names: dict[str, int]
    alias_collision_keys: int
    mapping_candidates: int
    missing_mapping_targets: int
    ingestion_statuses: dict[str, int]
    fatal_issues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class QualityInspector:
    CORE_TABLES = (
        "subjects",
        "characters",
        "persons",
        "episodes",
        "rel_subject_subject",
        "rel_subject_character",
        "rel_subject_person",
        "rel_person_character",
        "rel_person_person",
        "source_document",
        "community_document",
        "document_chunk",
        "document_role_section",
        "resolution_candidate",
        "entity_fts",
        "document_fts",
        "document_chunk_fts",
    )

    @classmethod
    def inspect(cls, database: Path) -> DatabaseQualityReport:
        database = database.resolve()
        with closing(sqlite3.connect(database)) as db:
            table_counts = {
                table: int(db.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
                for table in cls.CORE_TABLES
            }
            empty_names = {
                "subjects": int(
                    db.execute("SELECT count(*) FROM subjects WHERE trim(name)='' ").fetchone()[0]
                ),
                "characters": int(
                    db.execute("SELECT count(*) FROM characters WHERE trim(name)='' ").fetchone()[0]
                ),
                "persons": int(
                    db.execute("SELECT count(*) FROM persons WHERE trim(name)='' ").fetchone()[0]
                ),
            }
            alias_collision_keys = int(
                db.execute(
                    """
                    SELECT count(*) FROM (
                        SELECT normalized_alias
                        FROM entity_alias
                        GROUP BY normalized_alias
                        HAVING count(DISTINCT entity_id) > 1
                    )
                    """
                ).fetchone()[0]
            )
            mapping_candidates = table_counts["resolution_candidate"]
            missing_mapping_targets = int(
                db.execute(
                    """
                    SELECT count(*)
                    FROM resolution_candidate c
                    LEFT JOIN entity e ON e.id = c.right_entity_id
                    WHERE e.id IS NULL
                    """
                ).fetchone()[0]
            )
            ingestion_statuses = {
                str(status): int(count)
                for status, count in db.execute(
                    "SELECT status, count(*) FROM ingestion_run GROUP BY status"
                )
            }

        fatal: list[str] = []
        for table in ("subjects", "characters", "persons", "episodes"):
            if table_counts[table] == 0:
                fatal.append(f"核心表为空: {table}")
        for table, count in empty_names.items():
            if count:
                fatal.append(f"{table} 有 {count} 条空名称")
        not_loaded = sum(
            count for status, count in ingestion_statuses.items() if status != "loaded"
        )
        if not_loaded:
            fatal.append(f"有 {not_loaded} 个 ingestion_run 尚未 loaded")
        return DatabaseQualityReport(
            database=str(database),
            table_counts=table_counts,
            empty_names=empty_names,
            alias_collision_keys=alias_collision_keys,
            mapping_candidates=mapping_candidates,
            missing_mapping_targets=missing_mapping_targets,
            ingestion_statuses=ingestion_statuses,
            fatal_issues=tuple(fatal),
        )

    @staticmethod
    def assert_publishable(report: DatabaseQualityReport) -> None:
        if report.fatal_issues:
            raise RuntimeError("数据质量门禁失败: " + "; ".join(report.fatal_issues))

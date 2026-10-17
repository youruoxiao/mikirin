from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from mikirin.atomic import atomic_write_json
from mikirin.layout import ProjectLayout
from mikirin.schema import CONTENT_SCHEMA, CONTENT_SCHEMA_VERSION


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Publisher:
    def __init__(self, layout: ProjectLayout) -> None:
        self.layout = layout

    def create_empty_build(self, *, build_id: str, code_version: str) -> Path:
        self.layout.initialize()
        building = self.layout.building_db
        if building.exists():
            raise FileExistsError(f"旁路构建文件已存在，拒绝覆盖: {building}")
        try:
            with closing(sqlite3.connect(building)) as db, db:
                db.executescript(CONTENT_SCHEMA)
                db.execute(
                    "INSERT INTO schema_metadata(key, value) VALUES ('schema_version', ?)",
                    (str(CONTENT_SCHEMA_VERSION),),
                )
                now = utc_now()
                db.execute(
                    """
                    INSERT INTO build(id, started_at, completed_at, code_version, status)
                    VALUES (?, ?, ?, ?, 'validated')
                    """,
                    (build_id, now, now, code_version),
                )
            self.validate(building)
            return building
        except BaseException:
            building.unlink(missing_ok=True)
            raise

    @staticmethod
    def validate(path: Path) -> None:
        uri = f"{path.resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"SQLite 完整性检查失败: {integrity}")
            foreign_key_errors = db.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_errors:
                raise RuntimeError(f"外键检查失败: {foreign_key_errors[:5]}")

    def publish(self, *, version: str, keep: int = 3) -> Path:
        building = self.layout.building_db
        if not building.exists():
            raise FileNotFoundError(f"没有待发布构建: {building}")
        with closing(sqlite3.connect(building)) as db, db:
            db.execute(
                "UPDATE build SET completed_at = ?, status = 'published'",
                (utc_now(),),
            )
        self.validate(building)
        target = self.layout.db / f"content.v{version}.db"
        if target.exists():
            raise FileExistsError(f"版本已存在，拒绝覆盖: {target}")
        os.replace(building, target)
        atomic_write_json(
            self.layout.current_pointer,
            {
                "version": version,
                "database": target.name,
                "published_at": utc_now(),
                "schema_version": CONTENT_SCHEMA_VERSION,
            },
        )
        self._prune_versions(keep=keep, preserve=target)
        return target

    def current(self) -> Path | None:
        pointer = self.layout.current_pointer
        if not pointer.exists():
            return None
        with pointer.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        path = self.layout.db / payload["database"]
        if not path.exists():
            raise RuntimeError(f"current.json 指向不存在的数据库: {path}")
        return path

    def rollback(self, *, version: str) -> Path:
        target = self.layout.db / f"content.v{version}.db"
        if not target.exists():
            raise FileNotFoundError(f"回滚版本不存在: {target}")
        self.validate(target)
        atomic_write_json(
            self.layout.current_pointer,
            {
                "version": version,
                "database": target.name,
                "published_at": utc_now(),
                "schema_version": CONTENT_SCHEMA_VERSION,
                "rollback": True,
            },
        )
        return target

    def _prune_versions(self, *, keep: int, preserve: Path) -> None:
        if keep < 1:
            raise ValueError("keep 必须至少为 1")
        versions = sorted(
            self.layout.db.glob("content.v*.db"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for old in versions[keep:]:
            if old != preserve:
                old.unlink()

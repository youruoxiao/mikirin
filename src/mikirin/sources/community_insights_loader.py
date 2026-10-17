"""Load A3.3 community briefs and A3.6 discussion maps into a build."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


@dataclass(frozen=True, slots=True)
class CommunityInsightsLoadReport:
    counts: dict[str, int]


class CommunityInsightsLoader:
    """Transactional loader for the schema-v9 community insight tables."""

    @staticmethod
    def _connection(database_path: Path | str) -> sqlite3.Connection:
        db = sqlite3.connect(database_path)
        db.execute("PRAGMA foreign_keys = ON")
        return db

    @staticmethod
    def _theme_id(subject: str, episode: str | None, text: str, version: str) -> str:
        key = "\0".join((subject, episode or "", text, version))
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return "theme:" + uuid.uuid5(uuid.NAMESPACE_URL, digest).hex

    @staticmethod
    def _sync_report_fts(
        db: sqlite3.Connection, subject: str, record: dict[str, Any]
    ) -> None:
        db.execute(
            """INSERT INTO community_report_fts(
                subject_entity_id, reception_summary, strengths_json,
                weaknesses_json, discussion_points_json, key_episodes_json,
                notable_disagreements_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                subject,
                str(record.get("reception_summary", "")),
                _json(record.get("strengths", [])),
                _json(record.get("weaknesses", [])),
                _json(record.get("discussion_points", [])),
                _json(record.get("key_episodes", [])),
                _json(record.get("notable_disagreements", [])),
            ),
        )

    @staticmethod
    def _sync_theme_fts(
        db: sqlite3.Connection,
        subject: str,
        episode: str | None,
        rows: list[tuple[str, str, str]],
    ) -> None:
        for theme_id, theme_text, quotes in rows:
            db.execute(
                """INSERT INTO discussion_theme_fts(
                    id, subject_entity_id, episode_entity_id, theme_text, quotes_json
                ) VALUES (?, ?, ?, ?, ?)""",
                (theme_id, subject, episode, theme_text, quotes),
            )

    def load_community_briefs(
        self, staging_directory: Path | str, database_path: Path | str, build_id: str
    ) -> CommunityInsightsLoadReport:
        counts = {"loaded": 0, "skipped": 0, "dangling": 0}
        with closing(self._connection(database_path)) as db, db:
            # A staging run is an authoritative, replayable snapshot for this build.
            # Rebuild once instead of scanning FTS UNINDEXED anchor columns per subject.
            db.execute("DELETE FROM community_report_fts")
            db.execute("DELETE FROM community_report")
            for record in _rows(Path(staging_directory) / "briefs.jsonl"):
                subject = str(record.get("subject_entity_id", ""))
                if not subject or db.execute(
                    "SELECT 1 FROM subjects WHERE entity_id = ?", (subject,)
                ).fetchone() is None:
                    counts["dangling"] += 1
                    continue
                values = (
                    subject,
                    str(record.get("reception_summary", "")),
                    _json(record.get("strengths", [])),
                    _json(record.get("weaknesses", [])),
                    _json(record.get("discussion_points", [])),
                    _json(record.get("key_episodes", [])),
                    _json(record.get("notable_disagreements", [])),
                    _json(record.get("coverage", {})),
                    str(record.get("source_digest", "")),
                    str(record.get("sampling_policy_version", "")),
                    str(record.get("prompt_version", "brief-v1")),
                    str(record.get("model_fingerprint", "")),
                    str(record.get("run_id", "")),
                    build_id,
                )
                db.execute(
                    """INSERT INTO community_report(
                        subject_entity_id, reception_summary, strengths_json,
                        weaknesses_json, discussion_points_json, key_episodes_json,
                        notable_disagreements_json, coverage_json, source_digest,
                        sampling_policy_version, prompt_version, model_fingerprint,
                        run_id, build_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(subject_entity_id) DO UPDATE SET
                        reception_summary=excluded.reception_summary,
                        strengths_json=excluded.strengths_json,
                        weaknesses_json=excluded.weaknesses_json,
                        discussion_points_json=excluded.discussion_points_json,
                        key_episodes_json=excluded.key_episodes_json,
                        notable_disagreements_json=excluded.notable_disagreements_json,
                        coverage_json=excluded.coverage_json,
                        source_digest=excluded.source_digest,
                        sampling_policy_version=excluded.sampling_policy_version,
                        prompt_version=excluded.prompt_version,
                        model_fingerprint=excluded.model_fingerprint,
                        run_id=excluded.run_id, build_id=excluded.build_id""",
                    values,
                )
                self._sync_report_fts(db, subject, record)
                counts["loaded"] += 1
        return CommunityInsightsLoadReport(counts)

    def load_discussion_maps(
        self, staging_directory: Path | str, database_path: Path | str, build_id: str
    ) -> CommunityInsightsLoadReport:
        counts = {"loaded": 0, "skipped": 0, "dangling": 0}
        with closing(self._connection(database_path)) as db, db:
            # Full-run replacement also removes themes that disappeared after validation.
            db.execute("DELETE FROM discussion_theme_fts")
            db.execute("DELETE FROM discussion_theme")
            for record in _rows(Path(staging_directory) / "maps.jsonl"):
                subject = str(record.get("subject_entity_id", ""))
                episode_value = record.get("episode_entity_id")
                episode = None if episode_value is None else str(episode_value)
                valid_subject = db.execute(
                    "SELECT 1 FROM subjects WHERE entity_id = ?", (subject,)
                ).fetchone() is not None
                valid_episode = episode is None or db.execute(
                    "SELECT 1 FROM episodes WHERE entity_id = ? AND subject_entity_id = ?",
                    (episode, subject),
                ).fetchone() is not None
                if not valid_subject or not valid_episode:
                    counts["dangling"] += 1
                    continue
                version = str(record.get("prompt_version", "theme-v1"))
                model = str(record.get("extraction_model", ""))
                coverage = _json(record.get("coverage", {}))
                themes = record.get("themes", [])
                if not isinstance(themes, list):
                    counts["skipped"] += 1
                    continue
                fts_rows: list[tuple[str, str, str]] = []
                for theme in themes:
                    if not isinstance(theme, dict) or not str(theme.get("theme", "")).strip():
                        counts["skipped"] += 1
                        continue
                    text = str(theme["theme"])
                    aspect = str(theme.get("aspect", theme.get("aspect_kind", "other")))
                    quotes = theme.get("quotes", [])
                    if not isinstance(quotes, list):
                        counts["skipped"] += 1
                        continue
                    quotes_json = _json(quotes)
                    theme_id = self._theme_id(subject, episode, text, version)
                    db.execute(
                        """INSERT INTO discussion_theme(
                            id, subject_entity_id, episode_entity_id, theme_text,
                            aspect_kind, quotes_json, coverage_json, extraction_model,
                            extraction_version, build_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            theme_text=excluded.theme_text, aspect_kind=excluded.aspect_kind,
                            quotes_json=excluded.quotes_json, coverage_json=excluded.coverage_json,
                            extraction_model=excluded.extraction_model,
                            extraction_version=excluded.extraction_version, build_id=excluded.build_id""",
                        (theme_id, subject, episode, text, aspect, quotes_json, coverage, model, version, build_id),
                    )
                    fts_rows.append((theme_id, text, quotes_json))
                    counts["loaded"] += 1
                self._sync_theme_fts(db, subject, episode, fts_rows)
        return CommunityInsightsLoadReport(counts)

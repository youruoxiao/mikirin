"""Deterministic weekly memory audits. Only humans fill labels; no model grading."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path

from mikirin.atomic import atomic_write_json
from mikirin.run_lock import exclusive_process_lock


def _week(value: str | None) -> str:
    if value is None:
        year, week, _ = datetime.now(timezone.utc).date().isocalendar()
        return f"{year}-W{week:02d}"
    if not re.fullmatch(r"\d{4}-W\d{2}", value):
        raise ValueError("week must be YYYY-Www")
    date.fromisocalendar(int(value[:4]), int(value[-2:]), 1)
    return value


def _digest(cards: list[dict]) -> str:
    evidence = [{key: value for key, value in card.items() if key not in ("label", "notes")}
                for card in cards]
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def export_weekly_review(user_db: Path, output_path: Path, *, week: str | None = None) -> dict:
    """Export <=20 active cards and source evidence as JSON with blank human labels.

    Same week + same eligible snapshot produces the same sample and bytes. Reusing
    an existing output preserves human edits; another week's file is never overwritten.
    The database is opened read-only and never migrated by review operations.
    """
    week = _week(week)
    output_path = Path(output_path)
    with exclusive_process_lock(output_path.with_suffix(output_path.suffix + ".lock")):
        if output_path.exists():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if existing.get("week") != week:
                raise ValueError("output already belongs to another week; choose a new path")
            review_report(output_path)
            return existing
        with closing(sqlite3.connect(Path(user_db).resolve().as_uri() + "?mode=ro", uri=True)) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN")
            cards = [dict(row) for row in db.execute(
                "SELECT id AS note_id, content, keywords, context, entity_ids_json, "
                "source_session_id, valid_from, valid_until, created_at FROM memory_note "
                "WHERE valid_until IS NULL AND valid_from<=?", (datetime.now(timezone.utc).isoformat(),),
            )]
            population = len(cards)
            cards.sort(key=lambda card: (
                hashlib.sha256(f"memory-review-v1:{week}:{card['note_id']}".encode()).hexdigest(),
                card["note_id"],
            ))
            cards = cards[:20]
            has_jobs = db.execute("SELECT 1 FROM sqlite_master WHERE name="
                                  "'memory_settlement_job'").fetchone()
            for card in cards:
                session_id = card["source_session_id"]
                card["source_messages"] = [dict(row) for row in db.execute(
                    "SELECT role, content, id AS message_id FROM message WHERE session_id=? "
                    "ORDER BY created_at, rowid", (session_id,),
                )]
                card["settlement_inputs"] = []
                if has_jobs:
                    card["settlement_inputs"] = [
                        {"job_id": row["id"], "messages": json.loads(row["messages_json"])}
                        for row in db.execute("SELECT id, messages_json FROM memory_settlement_job "
                                              "WHERE session_id=? ORDER BY created_at, rowid",
                                              (session_id,))
                    ]
                card["label"] = ""
                card["notes"] = ""
        payload = {
            "schema": "mikirin-memory-review-v1", "week": week, "required_sample_size": 20,
            "eligible_population": population, "sample_size": len(cards),
            "labels": ["", "correct", "incorrect"], "cards": cards,
            "sample_digest": _digest(cards),
        }
        atomic_write_json(output_path, payload)
        return payload


def review_report(review_path: Path) -> dict:
    """Validate a human review; recommend upgrade only on a complete audit below 90%."""
    payload = json.loads(Path(review_path).read_text(encoding="utf-8"))
    if payload.get("schema") != "mikirin-memory-review-v1":
        raise ValueError("unsupported review schema")
    _week(payload["week"])
    cards = payload["cards"]
    if (not isinstance(cards, list) or len(cards) > 20
            or payload.get("sample_size") != len(cards)
            or payload.get("required_sample_size") != 20):
        raise ValueError("invalid review sample size")
    if len({card["note_id"] for card in cards}) != len(cards):
        raise ValueError("duplicate review card")
    if _digest(cards) != payload.get("sample_digest"):
        raise ValueError("review evidence changed; only label and notes may be edited")
    labels = [card.get("label", "") for card in cards]
    if any(label not in ("", "correct", "incorrect") for label in labels):
        raise ValueError("labels must be blank, correct, or incorrect")
    labeled = sum(label != "" for label in labels)
    correct = labels.count("correct")
    accuracy = correct / labeled if labeled else None
    complete = len(cards) == 20 and labeled == 20
    upgrade = complete and correct < 18
    return {
        "week": payload["week"], "sample_size": len(cards), "labeled": labeled,
        "correct": correct, "accuracy": accuracy, "complete": complete,
        "passed": complete and not upgrade, "upgrade_recommended": upgrade,
        "status": "passed" if complete and not upgrade else (
            "below_threshold" if upgrade else "pending_human_review"
        ),
    }

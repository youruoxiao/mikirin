"""topic 层构建：焦点作品卡（确定性，无 LLM，字节稳定吃前缀缓存）。

无锁定作品时输出固定空围栏；锁定后注入作品卡：
基本信息 + 各集评论热度（观剧地图雏形）。
"""

from __future__ import annotations

# topic layer extensions

import html
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from mikirin.conversation_focus import ConversationFocus

EMPTY_TOPIC = '<topic-context subject="none"/>'


# Optional community insight tables are handled here so older databases retain
# the original work card without requiring a schema migration.
_REPORT_ITEM_LIMIT = 5
_REPORT_ITEM_CHARS = 240
_SUMMARY_CHARS = 600
_QUOTE_LIMIT = 2
_QUOTE_CHARS = 280
_THEME_LIMIT = 3


def _safe(value: Any, limit: int | None = None) -> str:
    text = " ".join(str(value or "").split()).replace("```", "``\u200b`")
    if limit is not None:
        text = text[:limit]
    return html.escape(text, quote=True)


def _json_list(value: Any) -> list[Any]:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return parsed


def _item_text(item: Any) -> str:
    if isinstance(item, (str, int, float)):
        return str(item).strip()
    if not isinstance(item, dict):
        return ""
    text = next(
        (
            str(item[key]).strip()
            for key in ("point", "text", "summary", "reason", "description", "claim")
            if str(item.get(key, "")).strip()
        ),
        "",
    )
    if text and item.get("sort") is not None:
        return f"第{item['sort']}集：{text}"
    return text


def _has_table(db: sqlite3.Connection, name: str) -> bool:
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _episode_id(db: sqlite3.Connection, focus: ConversationFocus) -> str | None:
    if focus.active_episode_id is not None:
        row = db.execute(
            "SELECT entity_id FROM episodes WHERE entity_id=? AND subject_entity_id=?",
            (focus.active_episode_id, focus.active_subject_id),
        ).fetchone()
        if row is not None:
            return str(row[0])
    if focus.active_episode_sort is None:
        return None
    row = db.execute(
        "SELECT entity_id FROM episodes WHERE subject_entity_id=? AND sort=? "
        "ORDER BY entity_id LIMIT 1",
        (focus.active_subject_id, focus.active_episode_sort),
    ).fetchone()
    return None if row is None else str(row[0])


def _append_report(lines: list[str], db: sqlite3.Connection, subject_id: str) -> None:
    if not _has_table(db, "community_report"):
        return
    row = db.execute(
        "SELECT reception_summary, strengths_json, weaknesses_json, "
        "discussion_points_json, notable_disagreements_json, key_episodes_json "
        "FROM community_report WHERE subject_entity_id=?", (subject_id,)
    ).fetchone()
    if row is None:
        return
    summary = _safe(row["reception_summary"], _SUMMARY_CHARS)
    fields = (("strengths", row["strengths_json"]), ("weaknesses", row["weaknesses_json"]),
              ("discussion_points", row["discussion_points_json"]),
              ("notable_disagreements", row["notable_disagreements_json"]),
              ("key_episodes", row["key_episodes_json"]))
    values = []
    for name, value in fields:
        items = [
            _safe(text, _REPORT_ITEM_CHARS)
            for text in (_item_text(item) for item in _json_list(value))
            if text
        ][:_REPORT_ITEM_LIMIT]
        values.append((name, items))
    if not summary and not any(items for _, items in values):
        return
    lines.append("<community-report>")
    if summary:
        lines.append(f"reception_summary: {summary}")
    for name, items in values:
        if items:
            lines.append(f"{name}:")
            lines.extend(f"- {item}" for item in items)
    lines.append("</community-report>")


def _append_themes(lines: list[str], db: sqlite3.Connection, subject_id: str,
                   episode_id: str | None) -> None:
    if episode_id is None or not _has_table(db, "discussion_theme"):
        return
    rows = db.execute(
        "SELECT theme_text, aspect_kind, quotes_json FROM discussion_theme "
        "WHERE subject_entity_id=? AND episode_entity_id=? "
        "ORDER BY theme_text, aspect_kind, id LIMIT ?",
        (subject_id, episode_id, _THEME_LIMIT),
    ).fetchall()
    if not rows:
        return
    lines.append('<discussion-themes note="theme_text is a signpost; specific claims use quotes">')
    emitted = False
    for row in rows:
        theme = _safe(row["theme_text"], _REPORT_ITEM_CHARS)
        if not theme:
            continue
        emitted = True
        lines.append(f"theme_text: {theme}")
        lines.append(f"aspect_kind: {_safe(row['aspect_kind'], 80)}")
        quotes = []
        for item in _json_list(row["quotes_json"]):
            quote = item.get("quote") if isinstance(item, dict) else item
            if isinstance(quote, (str, int, float)) and str(quote).strip():
                quotes.append(_safe(quote, _QUOTE_CHARS))
        lines.extend(f"quote: {quote}" for quote in quotes[:_QUOTE_LIMIT])
    if emitted:
        lines.append("</discussion-themes>")


def build_topic_layer(database: Path, focus: ConversationFocus) -> str:
    if focus.is_empty or focus.active_subject_id is None:
        return EMPTY_TOPIC
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.row_factory = sqlite3.Row
        subject = db.execute(
            "SELECT s.name, s.name_cn, s.date, s.platform, sc.score, sc.rank "
            "FROM subjects s LEFT JOIN score_snapshot sc "
            "ON sc.subject_entity_id=s.entity_id WHERE s.entity_id=? "
            "ORDER BY sc.rowid DESC LIMIT 1", (focus.active_subject_id,)
        ).fetchone()
        if subject is None:
            return EMPTY_TOPIC
        heat = db.execute(
            "SELECT ep.sort, COUNT(*) AS n FROM community_document c "
            "JOIN episodes ep ON ep.entity_id=c.episode_entity_id "
            "WHERE c.subject_entity_id=? AND c.source_kind='bgm_ep_comment' "
            "GROUP BY ep.sort ORDER BY ep.sort", (focus.active_subject_id,)
        ).fetchall()
        episode_id = _episode_id(db, focus)
        lines = [
            f'<topic-context subject_id="{_safe(focus.active_subject_id)}" '
            f'episode_sort="{_safe(focus.active_episode_sort)}" confidence="{focus.confidence:.2f}">',
            f"subject: {_safe(subject['name_cn'] or subject['name'])}",
            f"date: {_safe(subject['date'] or 'unknown')}; score: {_safe(subject['score'] or '?')}; "
            f"rank: {_safe(subject['rank'] or '?')}",
        ]
        if heat:
            counts = [row["n"] for row in heat]
            average = sum(counts) / len(counts)
            hot = [f"{row['sort']:g}:{row['n']}" for row in heat if row["n"] > average * 1.5]
            if hot:
                lines.append("community_heat: " + ", ".join(hot))
        if focus.active_episode_sort is not None:
            lines.append(f"active_episode: {focus.active_episode_sort:g}")
        _append_report(lines, db, focus.active_subject_id)
        _append_themes(lines, db, focus.active_subject_id, episode_id)
        lines.append("</topic-context>")
        return "\n".join(lines)

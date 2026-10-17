"""人工复核队列的极简终端审阅。

逐条展示萌百/BGM 双侧卡片与 v4p 理由，按键裁决，
结果追加写入 data/manual/entity_resolution.jsonl（版本化人工构建输入）。
支持断点续审：已裁决的 candidate_id 自动跳过。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mikirin.layout import ProjectLayout

MANUAL_PATH = ("data", "manual", "entity_resolution.jsonl")


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(json.loads(line)["candidate_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _card(db: sqlite3.Connection, candidate_id: str) -> dict[str, Any]:
    db.row_factory = sqlite3.Row
    row = db.execute(
        """
        SELECT rc.left_entity_id, rc.right_entity_id,
               le.canonical_name AS lname, re.canonical_name AS rname
        FROM resolution_candidate rc
        JOIN entity le ON le.id = rc.left_entity_id
        JOIN entity re ON re.id = rc.right_entity_id
        WHERE rc.id = ?
        """,
        (candidate_id,),
    ).fetchone()
    card: dict[str, Any] = dict(row)
    facts = db.execute(
        "SELECT fact_kind, value_json FROM entity_fact WHERE entity_id=?",
        (row["left_entity_id"],),
    ).fetchall()
    card["left_facts"] = {f["fact_kind"]: f["value_json"] for f in facts}
    card["right_subjects"] = db.execute(
        """
        SELECT GROUP_CONCAT(COALESCE(NULLIF(s.name_cn,''), s.name), ' / ')
        FROM rel_subject_character rel
        JOIN subjects s ON s.entity_id = rel.subject_id
        WHERE rel.character_id = ?
        """,
        (row["right_entity_id"],),
    ).fetchone()[0]
    return card


def run_review(layout: ProjectLayout, queue_path: Path, min_collects: int) -> int:
    manual_path = layout.root.joinpath(*MANUAL_PATH)
    manual_path.parent.mkdir(parents=True, exist_ok=True)
    queue = [
        item
        for item in json.loads(queue_path.read_text(encoding="utf-8"))
        if item.get("collects", 0) >= min_collects
    ]
    done = _load_done(manual_path)
    pending = [item for item in queue if item["candidate_id"] not in done]
    print(f"待审 {len(pending)} / {len(queue)}（已完成 {len(done)}，min_collects={min_collects}）")
    if not pending:
        return 0

    from mikirin.publish import Publisher

    database = Publisher(layout).current()
    if database is None:
        raise FileNotFoundError("尚未发布可查询的 content.db")
    db = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    decided = 0
    with manual_path.open("a", encoding="utf-8") as out:
        for index, item in enumerate(pending, 1):
            card = _card(db, item["candidate_id"])
            facts = card["left_facts"]
            print("=" * 60)
            print(f"[{index}/{len(pending)}] 收藏 {item['collects']}")
            print(f"  萌百: {card['lname']}  作品:{facts.get('subjects','')[:60]}  声优:{facts.get('voice_actors','')[:40]}")
            print(f"  BGM : {card['rname']}  出演:{(card['right_subjects'] or '')[:60]}")
            print(f"  v4p : {item['v4p_decision']}({item['v4p_confidence']}) {item.get('v4p_reason','')[:80]}")
            choice = input("  s=同一角色 d=不同 u=拿不准 回车=跳过 q=退出 > ").strip().lower()
            if choice == "q":
                break
            if choice not in ("s", "d", "u"):
                continue
            record = {
                "candidate_id": item["candidate_id"],
                "left_entity_id": card["left_entity_id"],
                "right_entity_id": card["right_entity_id"],
                "decision": {"s": "same", "d": "different", "u": "uncertain"}[choice],
                "decided_by": "human",
                "decided_at": datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            decided += 1
    print(f"本次裁决 {decided} 条，写入 {manual_path}")
    return 0

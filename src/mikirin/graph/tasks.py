"""Offline heterogeneous task graphs over explicit credits, not overlap edges."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

TASK_GRAPH_VERSION = "heterogeneous-credits-v1"
HETEROGENEOUS_TASKS = {
    "voice_cast": ("voice_actor_role", "voiced_in", "character_in_work"),
    "staff_filmography": ("staff_credit",),
}


@dataclass
class CreditGraph:
    # A unit weight per distinct typed relationship: duplicate source assets
    # are provenance, not extra votes. Parallel relationship kinds may add up.
    edges: list[tuple[str, str, float, str]]
    node_types: dict[str, str]
    evidence: dict[tuple[str, str], list[dict[str, Any]]]


def load_credit_graph(db: sqlite3.Connection, task: str) -> CreditGraph:
    """Read source relations during an offline build; never used by search."""
    if task not in HETEROGENEOUS_TASKS:
        raise ValueError(f"unknown credit graph: {task}")
    typed: set[tuple[str, str, str]] = set()
    types: dict[str, str] = {}
    evidence: dict[tuple[str, str], dict[str, dict]] = {}

    def add(left: str, right: str, kind: str, record: dict) -> None:
        src, dst = sorted((left, right))
        typed.add((src, dst, kind))
        item = {"src_entity_id": src, "dst_entity_id": dst, "kind": kind, **record}
        evidence.setdefault((src, dst), {})[json.dumps(item, sort_keys=True)] = item

    if task == "staff_filmography":
        for subject, person, position, episodes, asset in db.execute(
            "SELECT subject_id,person_id,position,appear_eps_json,source_asset_id "
            "FROM rel_subject_person ORDER BY subject_id,person_id,position,source_asset_id"
        ):
            types[subject], types[person] = "subject", "person"
            add(person, subject, "staff_credit", {
                "source_table": "rel_subject_person", "source_asset_id": asset,
                "subject_entity_id": subject, "person_entity_id": person,
                "position": position, "appear_eps_json": episodes,
            })
    else:
        for person, subject, character, summary, asset in db.execute(
            "SELECT person_id,subject_id,character_id,summary,source_asset_id "
            "FROM rel_person_character ORDER BY person_id,subject_id,character_id,source_asset_id"
        ):
            types[person], types[subject], types[character] = "person", "subject", "character"
            record = {"source_table": "rel_person_character", "source_asset_id": asset,
                      "person_entity_id": person, "subject_entity_id": subject,
                      "character_entity_id": character, "summary": summary}
            add(person, character, "voice_actor_role", record)
            # Keep the actual credited work explicit. A shared character in a
            # sequel must not imply that every actor voiced every adaptation.
            add(person, subject, "voiced_in", record)
            add(character, subject, "character_in_work", record)
        for subject, character, role, order, asset in db.execute(
            "SELECT subject_id,character_id,role_type,rel_order,source_asset_id "
            "FROM rel_subject_character ORDER BY subject_id,character_id,role_type,source_asset_id"
        ):
            types[subject], types[character] = "subject", "character"
            add(character, subject, "character_in_work", {
                "source_table": "rel_subject_character", "source_asset_id": asset,
                "subject_entity_id": subject, "character_entity_id": character,
                "role_type": role, "rel_order": order,
            })
    return CreditGraph(
        [(src, dst, 1.0, kind) for src, dst, kind in sorted(typed)], types,
        {pair: [items[key] for key in sorted(items)] for pair, items in sorted(evidence.items())},
    )


def credit_path(graph: CreditGraph, parents: dict[str, str], seed: str, node: str) -> list[dict]:
    """A deterministic bounded-BFS witness, not a claim of direct employment."""
    path = []
    while node != seed:
        parent = parents[node]
        records = graph.evidence[tuple(sorted((parent, node)))]
        path.append({"from_entity_id": parent, "to_entity_id": node,
                     "source_records": records[:8], "source_record_count": len(records)})
        node = parent
    return list(reversed(path))

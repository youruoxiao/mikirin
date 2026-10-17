"""A2 实体合并构建：把 auto_same 裁决重放为实体合并。

萌百实体（``uuid5:character:*``）并入 BGM 权威实体（``bgm:character:*``，规范侧），
旧 ID 经 ``entity_redirect`` 永久可解析。合并是构建期重放——旁路构建上改写，
旧发布库不动。规格见 docs/SPEC-ROADMAP.md §1（A2）；ADR-1/2/3/7/8 不可偏离。

三个入口：
- :func:`load_merge_map` 构造并校验合并映射（§1.2，优先级 人工 > pro > flash > 规则）；
- :func:`apply_merges` 在构建库上幂等重放合并（§1.3）；
- :func:`verify_golden` 跑硬门禁与 DQ（§1.5），不过不发布。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


# 规范侧来源前缀（BGM 权威实体）与萌百 provisional 前缀。
BGM_ENTITY_PREFIX = "bgm:"
PROVISIONAL_ENTITY_PREFIX = "uuid5:"

# DQ 基线（§1.5）：auto_same 期望数与规范实体携带萌百来源别名/事实的下限。
BASELINE_REDIRECT_COUNT = 22_027
MIN_CANONICAL_WITH_MOEGIRL = 21_900


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: Any) -> str:
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


class MergeConflictError(RuntimeError):
    """合并映射构造期命中硬冲突（一对多 / redirect 链成环），拒绝发布。"""

    def __init__(self, message: str, conflicts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.conflicts = conflicts


class GoldenGateError(RuntimeError):
    """黄金集硬门禁失败（人工裁决未被正确重放），拒绝发布。"""

    def __init__(self, message: str, failures: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.failures = failures


@dataclass(frozen=True, slots=True)
class MergePair:
    old_entity_id: str
    canonical_entity_id: str
    decision_source: str
    confidence: float
    decided_at: str
    candidate_id: str | None = None
    matched_evidence: tuple[str, ...] = ()
    conflicting_evidence: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MergePlan:
    merges: dict[str, MergePair]
    candidate_statuses: dict[str, str]
    vetoes: tuple[dict[str, Any], ...]
    awareness: tuple[dict[str, Any], ...]
    unresolved_candidates: tuple[str, ...]

    @property
    def merge_count(self) -> int:
        return len(self.merges)


@dataclass(frozen=True, slots=True)
class MergeReport:
    merges_applied: int
    redirects: int
    decisions: int
    aliases_moved: int
    alias_conflicts: int
    facts_moved: int
    fact_conflicts: int
    candidate_status_counts: dict[str, int]
    alias_conflict_log: str | None
    fact_conflict_log: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GoldenReport:
    human_checked: int
    human_passed: int
    redirect_count: int
    merged_entity_count: int
    canonical_with_moegirl: int
    baseline_redirect_count: int
    baseline_delta: int
    canonical_with_moegirl_floor: int
    canonical_with_moegirl_ok: bool
    llm_agreement: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_manual_decisions(path: Path) -> list[dict[str, Any]]:
    """读取人工裁决（最高优先级）。每行含 candidate_id/左右实体/decision/decided_by。

    同一 candidate_id 多次裁决时后者覆盖前者——纠错通道：追加一条新裁决即可
    推翻旧裁决，下游（合并映射、verify_golden）只验最新一条。
    """
    if not path.exists():
        return []
    by_candidate: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for record in _rows(path):
        decision = str(record.get("decision", "")).strip().lower()
        if decision not in ("same", "different"):
            continue
        item = {
            "candidate_id": record.get("candidate_id"),
            "left_entity_id": str(record["left_entity_id"]),
            "right_entity_id": str(record["right_entity_id"]),
            "decision": decision,
            "decided_by": str(record.get("decided_by", "human")),
            "decided_at": str(record.get("decided_at") or _utc_now()),
        }
        candidate_id = item["candidate_id"]
        if candidate_id:
            by_candidate[candidate_id] = item  # 后写覆盖：最新裁决生效
        else:
            anonymous.append(item)
    return list(by_candidate.values()) + anonymous


def _load_compiled(path: Path) -> dict[str, dict[str, Any]]:
    compiled: dict[str, dict[str, Any]] = {}
    for record in _rows(path):
        compiled[str(record["candidate_id"])] = record
    return compiled


def _load_candidate_index(db: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    rows = db.execute(
        """
        SELECT id, left_entity_id, right_entity_id, confidence, decision_source
        FROM resolution_candidate
        """
    )
    for candidate_id, left, right, confidence, source in rows:
        index[str(candidate_id)] = {
            "left_entity_id": str(left),
            "right_entity_id": str(right),
            "confidence": float(confidence),
            "decision_source": str(source),
        }
    return index


def _orient(old: str, canonical: str) -> tuple[str, str]:
    """确保 old 是萌百 provisional 侧、canonical 是 BGM 规范侧。

    人工裁决左右顺序固定（left=萌百/right=BGM），但 compiled 经 DB 解析得到的
    left/right 也遵循同一约定；这里做一次防御性归一，万一顺序颠倒仍能正确合并。
    """
    if old.startswith(BGM_ENTITY_PREFIX) and canonical.startswith(PROVISIONAL_ENTITY_PREFIX):
        return canonical, old
    return old, canonical


def load_merge_map(
    database: Path,
    *,
    resolution_compiled: Path,
    manual_resolution: Path | None = None,
) -> MergePlan:
    """构造合并映射并执行冲突检测（§1.2）。

    硬冲突（命中即抛 :class:`MergeConflictError`，拒绝发布）：
    - 同一萌百实体合并到 ≥2 个不同 BGM 实体；
    - redirect 链成环（A→B→…→A）。
    redirect 链（非环）归一到终点；N:1 且全非人工仅列入 awareness 供知情。
    """
    compiled = _load_compiled(resolution_compiled)
    manual = load_manual_decisions(manual_resolution) if manual_resolution else []
    with closing(sqlite3.connect(database)) as db:
        candidate_index = _load_candidate_index(db)

    now = _utc_now()
    proposed: list[MergePair] = []
    candidate_statuses: dict[str, str] = {}
    vetoes: list[dict[str, Any]] = []
    unresolved: list[str] = []

    # 1) compiled 裁决：auto_same 进合并提议（先不折叠），其余落候选终态。
    for candidate_id, record in compiled.items():
        final_decision = str(record.get("final_decision"))
        if final_decision == "auto_same":
            source_record = candidate_index.get(candidate_id)
            if source_record is None:
                unresolved.append(candidate_id)
                continue
            old, canonical = _orient(
                source_record["left_entity_id"], source_record["right_entity_id"]
            )
            proposed.append(
                MergePair(
                    old_entity_id=old,
                    canonical_entity_id=canonical,
                    decision_source=str(record.get("decision_source", "rule")),
                    confidence=float(record.get("confidence") or source_record["confidence"]),
                    decided_at=now,
                    candidate_id=candidate_id,
                    matched_evidence=tuple(record.get("matched_evidence") or ()),
                    conflicting_evidence=tuple(record.get("conflicting_evidence") or ()),
                )
            )
            candidate_statuses[candidate_id] = "merged"
        elif final_decision in ("auto_different",):
            candidate_statuses[candidate_id] = "rejected"
            vetoes.append(
                {"candidate_id": candidate_id, "reason": "auto_different", "by": "compiled"}
            )
        elif final_decision in ("dropped_sparse", "dropped"):
            candidate_statuses[candidate_id] = "dropped"
        else:  # review / uncertain / 低置信 same（ADR-7：低置信 same 一律不合并）
            candidate_statuses[candidate_id] = "review"

    # 2) 人工裁决：same 强制加入、different 强制否决（纠错通道，最高优先级）。
    manual_veto_pairs: set[tuple[str, str]] = set()
    for record in manual:
        candidate_id = record["candidate_id"]
        old, canonical = _orient(record["left_entity_id"], record["right_entity_id"])
        if record["decision"] == "same":
            proposed.append(
                MergePair(
                    old_entity_id=old,
                    canonical_entity_id=canonical,
                    decision_source="human",
                    confidence=1.0,
                    decided_at=record["decided_at"],
                    candidate_id=candidate_id,
                )
            )
            if candidate_id:
                candidate_statuses[candidate_id] = "merged"
        else:  # different
            manual_veto_pairs.add((old, canonical))
            vetoes.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "human_different",
                    "by": record["decided_by"],
                    "old_entity_id": old,
                    "canonical_entity_id": canonical,
                }
            )
            if candidate_id:
                candidate_statuses[candidate_id] = "rejected"

    # 3) 人工否决优先（§1.2 规则2 先于规则3）：先剔除被否决的对，再查冲突。
    surviving = [
        pair
        for pair in proposed
        if (pair.old_entity_id, pair.canonical_entity_id) not in manual_veto_pairs
    ]

    # 4) 冲突检测（a）：同一萌百实体 → ≥2 个不同 BGM 实体，命中即拒绝发布。
    one_to_many = _detect_one_to_many(surviving)
    if one_to_many:
        raise MergeConflictError(
            "同一萌百实体合并到多个 BGM 实体，拒绝发布:\n"
            + json.dumps(one_to_many, ensure_ascii=False, indent=2),
            one_to_many,
        )

    # 5) 折叠成映射：已无一对多，每个 old 至多一个 canonical；同对多源时人工优先。
    collapsed: dict[str, MergePair] = {}
    for pair in surviving:
        existing = collapsed.get(pair.old_entity_id)
        if existing is None or (
            pair.decision_source == "human" and existing.decision_source != "human"
        ):
            collapsed[pair.old_entity_id] = pair

    # 6) redirect 链归一（b）：canonical 自身也是 old 时跟随到终点；成环即拒绝。
    merges, chains = _normalize_chains(collapsed)
    if chains["cycles"]:
        raise MergeConflictError(
            "redirect 链成环，拒绝发布:\n"
            + json.dumps(chains["cycles"], ensure_ascii=False, indent=2),
            chains["cycles"],
        )

    # 7) awareness（c）：同一 BGM 实体接收 ≥2 个萌百实体且全非人工——列出供知情。
    awareness = _detect_fan_in(merges)

    return MergePlan(
        merges=merges,
        candidate_statuses=candidate_statuses,
        vetoes=tuple(vetoes),
        awareness=tuple(awareness),
        unresolved_candidates=tuple(sorted(unresolved)),
    )


def _detect_one_to_many(pairs: list[MergePair]) -> list[dict[str, Any]]:
    """存活合并对里，同一萌百实体指向 ≥2 个不同 BGM 实体即硬冲突。

    入参已是剔除人工否决后的对（§1.2 规则2 先于规则3）。
    """
    targets: dict[str, set[str]] = {}
    sources: dict[str, set[str]] = {}
    for pair in pairs:
        targets.setdefault(pair.old_entity_id, set()).add(pair.canonical_entity_id)
        sources.setdefault(pair.old_entity_id, set()).add(pair.decision_source)
    return [
        {
            "old_entity_id": old,
            "canonical_targets": sorted(canonicals),
            "sources": sorted(sources[old]),
        }
        for old, canonicals in sorted(targets.items())
        if len(canonicals) > 1
    ]


def _normalize_chains(
    proposed: dict[str, MergePair],
) -> tuple[dict[str, MergePair], dict[str, list[Any]]]:
    target_of = {old: pair.canonical_entity_id for old, pair in proposed.items()}
    normalized: dict[str, MergePair] = {}
    cycles: list[list[str]] = []
    for old, pair in proposed.items():
        seen = [old]
        canonical = pair.canonical_entity_id
        while canonical in target_of:
            if canonical in seen:
                cycles.append(seen + [canonical])
                canonical = ""
                break
            seen.append(canonical)
            canonical = target_of[canonical]
        if canonical:
            normalized[old] = MergePair(
                old_entity_id=pair.old_entity_id,
                canonical_entity_id=canonical,
                decision_source=pair.decision_source,
                confidence=pair.confidence,
                decided_at=pair.decided_at,
                candidate_id=pair.candidate_id,
                matched_evidence=pair.matched_evidence,
                conflicting_evidence=pair.conflicting_evidence,
            )
    return normalized, {"cycles": cycles}


def _detect_fan_in(merges: dict[str, MergePair]) -> list[dict[str, Any]]:
    by_canonical: dict[str, list[MergePair]] = {}
    for pair in merges.values():
        by_canonical.setdefault(pair.canonical_entity_id, []).append(pair)
    awareness: list[dict[str, Any]] = []
    for canonical, pairs in sorted(by_canonical.items()):
        if len(pairs) >= 2 and all(pair.decision_source != "human" for pair in pairs):
            awareness.append(
                {
                    "canonical_entity_id": canonical,
                    "merged_olds": sorted(pair.old_entity_id for pair in pairs),
                    "decision_sources": sorted({pair.decision_source for pair in pairs}),
                }
            )
    return awareness


def apply_merges(
    database: Path,
    plan: MergePlan,
    *,
    build_id: str,
    rejected_dir: Path,
    quality_dir: Path,
) -> MergeReport:
    """在构建库上幂等重放合并（§1.3）。每对事务内执行，重跑结果不变。"""
    rejected_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)
    alias_conflict_log = rejected_dir / "merge-alias-conflicts.jsonl"
    fact_conflict_log = quality_dir / f"fact-conflicts-{build_id}.jsonl"

    alias_conflicts: list[dict[str, Any]] = []
    fact_conflicts: list[dict[str, Any]] = []
    counters = {
        "merges_applied": 0,
        "redirects": 0,
        "decisions": 0,
        "aliases_moved": 0,
        "facts_moved": 0,
    }

    with closing(sqlite3.connect(database)) as db, db:
        db.execute("PRAGMA foreign_keys = ON")
        for old, pair in sorted(plan.merges.items()):
            canonical = pair.canonical_entity_id
            counters["aliases_moved"] += _repoint_aliases(db, old, canonical, alias_conflicts)
            counters["facts_moved"] += _repoint_facts(db, old, canonical, fact_conflicts)
            db.execute(
                """
                UPDATE source_entity
                SET entity_id=?, mapping_status='resolved', mapping_confidence=1.0
                WHERE id=? AND source='moegirl_dataset'
                """,
                (canonical, old),
            )
            reason = f"auto_same:{pair.decision_source}"
            before = db.total_changes
            db.execute(
                """
                INSERT OR IGNORE INTO entity_redirect(
                    old_entity_id, canonical_entity_id, reason, decided_at
                ) VALUES (?, ?, ?, ?)
                """,
                (old, canonical, reason, pair.decided_at),
            )
            counters["redirects"] += db.total_changes - before
            decision_id = "decision:" + uuid.uuid5(
                uuid.NAMESPACE_URL, f"{old}\0{canonical}\0merge"
            ).hex
            evidence = {
                "candidate_id": pair.candidate_id,
                "confidence": pair.confidence,
                "matched_evidence": list(pair.matched_evidence),
                "conflicting_evidence": list(pair.conflicting_evidence),
            }
            before = db.total_changes
            db.execute(
                """
                INSERT OR IGNORE INTO resolution_decision(
                    id, left_source_entity_id, right_source_entity_id, decision,
                    evidence_json, decision_source, decided_at
                ) VALUES (?, ?, ?, 'same', ?, ?, ?)
                """,
                (
                    decision_id,
                    old,
                    canonical,
                    _json(evidence),
                    pair.decision_source,
                    pair.decided_at,
                ),
            )
            counters["decisions"] += db.total_changes - before
            db.execute("UPDATE entity SET status='merged' WHERE id=?", (old,))
            counters["merges_applied"] += 1

        status_counts: dict[str, int] = {}
        for candidate_id, status in sorted(plan.candidate_statuses.items()):
            status_counts[status] = status_counts.get(status, 0) + 1
            db.execute(
                "UPDATE resolution_candidate SET status=? WHERE id=?",
                (status, candidate_id),
            )

    _write_jsonl(alias_conflict_log, alias_conflicts)
    _write_jsonl(fact_conflict_log, fact_conflicts)

    return MergeReport(
        merges_applied=counters["merges_applied"],
        redirects=counters["redirects"],
        decisions=counters["decisions"],
        aliases_moved=counters["aliases_moved"],
        alias_conflicts=len(alias_conflicts),
        facts_moved=counters["facts_moved"],
        fact_conflicts=len(fact_conflicts),
        candidate_status_counts=status_counts,
        alias_conflict_log=str(alias_conflict_log) if alias_conflicts else None,
        fact_conflict_log=str(fact_conflict_log) if fact_conflicts else None,
    )


def _repoint_aliases(
    db: sqlite3.Connection, old: str, canonical: str, log: list[dict[str, Any]]
) -> int:
    """把 old 的别名迁到 canonical；同 (normalized_alias, source_kind, build_id)
    冲突保留置信度高者，被替换行落 merge-alias-conflicts.jsonl。返回迁移条数。"""
    conflict_rows = db.execute(
        """
        SELECT oa.id
        FROM entity_alias oa
        JOIN entity_alias ca
            ON ca.entity_id=?
           AND ca.normalized_alias=oa.normalized_alias
           AND ca.source_kind=oa.source_kind
           AND ca.build_id IS oa.build_id
        WHERE oa.entity_id=?
        """,
        (canonical, old),
    ).fetchall()
    for (old_alias_id,) in conflict_rows:
        old_row = _fetch_alias(db, old_alias_id)
        canonical_row = db.execute(
            """
            SELECT id, entity_id, alias_text, normalized_alias, source_kind, status,
                   confidence, build_id
            FROM entity_alias
            WHERE entity_id=? AND normalized_alias=? AND source_kind=?
              AND build_id IS ?
            """,
            (
                canonical,
                old_row["normalized_alias"],
                old_row["source_kind"],
                old_row["build_id"],
            ),
        ).fetchone()
        if canonical_row is None:
            continue
        loser, winner = _pick_loser(old_row, _alias_from_row(canonical_row))
        db.execute("DELETE FROM entity_alias WHERE id=?", (loser["id"],))
        log.append(
            {
                "old_entity_id": old,
                "canonical_entity_id": canonical,
                "normalized_alias": loser["normalized_alias"],
                "source_kind": loser["source_kind"],
                "replaced_alias": loser,
                "kept_alias": winner,
            }
        )
    before = db.total_changes
    db.execute("UPDATE entity_alias SET entity_id=? WHERE entity_id=?", (canonical, old))
    return db.total_changes - before


def _repoint_facts(
    db: sqlite3.Connection, old: str, canonical: str, log: list[dict[str, Any]]
) -> int:
    """把 old 的事实迁到 canonical。UNIQUE(entity_id, fact_kind, source_asset_id)
    在共享 asset 的 N:1 合并下会真冲突：按 ADR-2 保留置信度高者入表，矛盾对双方
    完整落 fact-conflicts 供人工抽查（证据比结论长寿）。返回迁移条数。"""
    conflict_rows = db.execute(
        """
        SELECT ef.id
        FROM entity_fact ef
        JOIN entity_fact cf
            ON cf.entity_id=?
           AND cf.fact_kind=ef.fact_kind
           AND cf.source_asset_id=ef.source_asset_id
        WHERE ef.entity_id=?
        """,
        (canonical, old),
    ).fetchall()
    for (old_fact_id,) in conflict_rows:
        old_row = _fetch_fact(db, old_fact_id)
        canonical_row = db.execute(
            """
            SELECT id, entity_id, fact_kind, value_json, source_asset_id, confidence
            FROM entity_fact
            WHERE entity_id=? AND fact_kind=? AND source_asset_id=?
            """,
            (canonical, old_row["fact_kind"], old_row["source_asset_id"]),
        ).fetchone()
        if canonical_row is None:
            continue
        loser, winner = _pick_loser(old_row, _fact_from_row(canonical_row))
        db.execute("DELETE FROM entity_fact WHERE id=?", (loser["id"],))
        log.append(
            {
                "old_entity_id": old,
                "canonical_entity_id": canonical,
                "fact_kind": loser["fact_kind"],
                "kept": winner,
                "dropped": loser,
            }
        )
    before = db.total_changes
    db.execute("UPDATE entity_fact SET entity_id=? WHERE entity_id=?", (canonical, old))
    return db.total_changes - before


def _fetch_alias(db: sqlite3.Connection, alias_id: str) -> dict[str, Any]:
    row = db.execute(
        """
        SELECT id, entity_id, alias_text, normalized_alias, source_kind, status,
               confidence, build_id
        FROM entity_alias WHERE id=?
        """,
        (alias_id,),
    ).fetchone()
    return _alias_from_row(row)


def _alias_from_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "entity_id": row[1],
        "alias_text": row[2],
        "normalized_alias": row[3],
        "source_kind": row[4],
        "status": row[5],
        "confidence": float(row[6]),
        "build_id": row[7],
    }


def _fetch_fact(db: sqlite3.Connection, fact_id: str) -> dict[str, Any]:
    row = db.execute(
        """
        SELECT id, entity_id, fact_kind, value_json, source_asset_id, confidence
        FROM entity_fact WHERE id=?
        """,
        (fact_id,),
    ).fetchone()
    return _fact_from_row(row)


def _fact_from_row(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "entity_id": row[1],
        "fact_kind": row[2],
        "value_json": row[3],
        "source_asset_id": row[4],
        "confidence": float(row[5]),
    }


def _pick_loser(
    old_row: dict[str, Any], canonical_row: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """置信度高者入表；平手保留规范侧（canonical）既有行，确定性可复现。"""
    if old_row["confidence"] > canonical_row["confidence"]:
        return canonical_row, old_row
    return old_row, canonical_row


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def verify_golden(
    database: Path,
    *,
    manual_resolution: Path,
    plan: MergePlan,
    baseline_redirect_count: int = BASELINE_REDIRECT_COUNT,
    min_canonical_with_moegirl: int = MIN_CANONICAL_WITH_MOEGIRL,
    llm_labels: list[dict[str, Any]] | None = None,
) -> GoldenReport:
    """硬门禁 + DQ（§1.5）。人工裁决任一不过抛 :class:`GoldenGateError`。"""
    manual = load_manual_decisions(manual_resolution)
    failures: list[dict[str, Any]] = []
    with closing(sqlite3.connect(database)) as db:
        for record in manual:
            old, canonical = _orient(record["left_entity_id"], record["right_entity_id"])
            if record["decision"] == "same":
                row = db.execute(
                    """
                    SELECT 1 FROM entity_redirect
                    WHERE old_entity_id=? AND canonical_entity_id=?
                    """,
                    (old, canonical),
                ).fetchone()
                if row is None:
                    failures.append(
                        {
                            "candidate_id": record["candidate_id"],
                            "expectation": "same → redirect 必须存在",
                            "old_entity_id": old,
                            "canonical_entity_id": canonical,
                        }
                    )
            else:
                row = db.execute(
                    """
                    SELECT 1 FROM entity_redirect
                    WHERE (old_entity_id=? AND canonical_entity_id=?)
                       OR (old_entity_id=? AND canonical_entity_id=?)
                    """,
                    (old, canonical, canonical, old),
                ).fetchone()
                if row is not None:
                    failures.append(
                        {
                            "candidate_id": record["candidate_id"],
                            "expectation": "different → 两实体间不得存在 redirect",
                            "old_entity_id": old,
                            "canonical_entity_id": canonical,
                        }
                    )

        redirect_count = int(db.execute("SELECT count(*) FROM entity_redirect").fetchone()[0])
        merged_entity_count = int(
            db.execute("SELECT count(*) FROM entity WHERE status='merged'").fetchone()[0]
        )
        # 规范实体 = redirect 目标（BGM 侧）；未合并的萌百限定角色（ADR-8）不计入。
        canonical_with_moegirl = int(
            db.execute(
                """
                SELECT count(DISTINCT carrier.entity_id) FROM (
                    SELECT entity_id FROM entity_alias WHERE source_kind='moegirl_dataset'
                    UNION
                    SELECT ef.entity_id FROM entity_fact ef
                    JOIN source_asset sa ON sa.id=ef.source_asset_id
                    JOIN ingestion_run ir ON ir.id=sa.ingestion_run_id
                    WHERE ir.source='moegirl_dataset'
                ) carrier
                JOIN entity_redirect r ON r.canonical_entity_id=carrier.entity_id
                """
            ).fetchone()[0]
        )

    if failures:
        raise GoldenGateError(
            "黄金集硬门禁失败（人工裁决未正确重放）:\n"
            + json.dumps(failures, ensure_ascii=False, indent=2),
            failures,
        )

    consistency: list[dict[str, Any]] = []
    if redirect_count != merged_entity_count:
        consistency.append(
            {
                "check": "redirect 数 == entity merged 数",
                "redirect_count": redirect_count,
                "merged_entity_count": merged_entity_count,
            }
        )
    if redirect_count != plan.merge_count:
        consistency.append(
            {
                "check": "redirect 数 == 合并映射对数",
                "redirect_count": redirect_count,
                "merge_count": plan.merge_count,
            }
        )
    # 绝对数字（22,027 / 21,900）是基于不完整假设的估算，降为监控指标（见下）；
    # 硬门禁只查内部一致性，防止“合并跑了但萌百数据没挂上”这类真故障。
    if redirect_count > 0 and canonical_with_moegirl == 0:
        consistency.append(
            {
                "check": "有合并却无规范实体携带萌百来源数据",
                "redirect_count": redirect_count,
                "canonical_with_moegirl": canonical_with_moegirl,
            }
        )
    if canonical_with_moegirl > redirect_count:
        consistency.append(
            {
                "check": "携带萌百数据的规范实体多于 redirect 数",
                "canonical_with_moegirl": canonical_with_moegirl,
                "redirect_count": redirect_count,
            }
        )
    if consistency:
        raise GoldenGateError(
            "DQ 一致性门禁失败:\n" + json.dumps(consistency, ensure_ascii=False, indent=2),
            consistency,
        )

    return GoldenReport(
        human_checked=len(manual),
        human_passed=len(manual),
        redirect_count=redirect_count,
        merged_entity_count=merged_entity_count,
        canonical_with_moegirl=canonical_with_moegirl,
        baseline_redirect_count=baseline_redirect_count,
        baseline_delta=redirect_count - baseline_redirect_count,
        canonical_with_moegirl_floor=min_canonical_with_moegirl,
        canonical_with_moegirl_ok=canonical_with_moegirl >= min_canonical_with_moegirl,
        llm_agreement=_llm_agreement(plan, llm_labels),
    )


def _llm_agreement(
    plan: MergePlan, labels: list[dict[str, Any]] | None
) -> float | None:
    """监控指标（不阻断）：与 LLM 标注集的一致率。无标注集返回 None。"""
    if not labels:
        return None
    agreed = 0
    total = 0
    for label in labels:
        candidate_id = label.get("candidate_id")
        expected = str(label.get("decision", "")).lower()
        if candidate_id is None or expected not in ("same", "different"):
            continue
        total += 1
        actual_status = plan.candidate_statuses.get(candidate_id)
        actual = "same" if actual_status == "merged" else "different"
        if actual == expected:
            agreed += 1
    return agreed / total if total else None

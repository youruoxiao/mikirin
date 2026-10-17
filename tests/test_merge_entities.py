"""A2 实体合并构建测试（纯标准库，不联网）。

覆盖 SPEC-ROADMAP §1.6 要求的六类场景：迷你库全字段验证、幂等重跑、
一人并两物拒绝、人工否决优先、redirect 链归一、查询跟随 redirect；
另含 N:1 事实冲突落盘与黄金集门禁负例。
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unicodedata
import unittest
import uuid
from contextlib import closing
from pathlib import Path

from mikirin.layout import ProjectLayout
from mikirin.merge_entities import (
    GoldenGateError,
    MergeConflictError,
    apply_merges,
    load_merge_map,
    verify_golden,
)
from mikirin.publish import Publisher
from mikirin.query import KnowledgeQueries
from mikirin.search_index import SearchIndexer


NOW = "2026-07-28T00:00:00+00:00"
BUILD_ID = "build-test"


def _norm(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _candidate_id(left: str, right: str) -> str:
    return "candidate:" + uuid.uuid5(
        uuid.NAMESPACE_URL, f"{left}\0{right}\0same_character"
    ).hex


def _setup_db(layout: ProjectLayout) -> Path:
    database = Publisher(layout).create_empty_build(
        build_id=BUILD_ID, code_version="test"
    )
    with closing(sqlite3.connect(database)) as db, db:
        db.execute(
            "INSERT INTO ingestion_run(id, source, started_at, source_version, "
            "parser_version, status) VALUES ('ingest:bgm', 'bgm', ?, 'v1', 'p', 'loaded')",
            (NOW,),
        )
        db.execute(
            "INSERT INTO source_asset(id, ingestion_run_id, local_path, content_hash, "
            "fetched_at) VALUES ('asset:bgm', 'ingest:bgm', '/bgm', 'sha256:bgm', ?)",
            (NOW,),
        )
        db.execute(
            "INSERT INTO ingestion_run(id, source, started_at, source_version, "
            "parser_version, status) VALUES "
            "('ingest:moegirl', 'moegirl_dataset', ?, 'v1', 'p', 'loaded')",
            (NOW,),
        )
        db.execute(
            "INSERT INTO source_asset(id, ingestion_run_id, local_path, content_hash, "
            "fetched_at) VALUES ('asset:moegirl', 'ingest:moegirl', '/moegirl', "
            "'sha256:moegirl', ?)",
            (NOW,),
        )
    return database


def _add_bgm_character(
    db: sqlite3.Connection, entity_id: str, name: str, aliases: tuple[str, ...] = ()
) -> None:
    db.execute(
        "INSERT INTO entity VALUES (?, 'character', ?, 'active')", (entity_id, name)
    )
    db.execute(
        "INSERT INTO source_entity VALUES (?, 'bgm', 'character', ?, ?, ?, NULL, "
        "'resolved', 1.0, 'asset:bgm')",
        (entity_id, entity_id.split(":")[-1], entity_id, name),
    )
    for alias in aliases:
        normalized = _norm(alias)
        alias_id = "alias:" + uuid.uuid5(
            uuid.NAMESPACE_URL, f"{entity_id}\0{normalized}\0bgm_archive"
        ).hex
        db.execute(
            "INSERT OR IGNORE INTO entity_alias VALUES (?, ?, ?, ?, NULL, "
            "'bgm_archive', 'active', 1.0, 'asset:bgm', ?, 'upstream')",
            (alias_id, entity_id, alias, normalized, BUILD_ID),
        )


def _add_moegirl_character(
    db: sqlite3.Connection,
    entity_id: str,
    name: str,
    *,
    aliases: tuple[str, ...] = (),
    gender: str | None = None,
    canonical: str | None = None,
) -> str | None:
    """插入萌百 provisional 实体；给定 canonical 时同步建 resolution_candidate。"""
    db.execute(
        "INSERT INTO entity VALUES (?, 'character', ?, 'provisional')", (entity_id, name)
    )
    db.execute(
        "INSERT INTO source_entity VALUES (?, 'moegirl_dataset', 'character', ?, ?, "
        "?, NULL, 'unresolved', 0.0, 'asset:moegirl')",
        (entity_id, entity_id.split(":")[-1], entity_id, name),
    )
    for alias in aliases:
        normalized = _norm(alias)
        alias_id = "alias:" + uuid.uuid5(
            uuid.NAMESPACE_URL, f"{entity_id}\0{normalized}\0moegirl_dataset"
        ).hex
        db.execute(
            "INSERT OR IGNORE INTO entity_alias VALUES (?, ?, ?, ?, NULL, "
            "'moegirl_dataset', 'candidate', 0.8, 'asset:moegirl', ?, 'upstream')",
            (alias_id, entity_id, alias, normalized, BUILD_ID),
        )
    if gender is not None:
        fact_id = "fact:" + uuid.uuid5(
            uuid.NAMESPACE_URL, f"{entity_id}\0gender\0asset:moegirl"
        ).hex
        db.execute(
            "INSERT INTO entity_fact VALUES (?, ?, 'gender', ?, 'asset:moegirl', 0.8)",
            (fact_id, entity_id, json.dumps(gender, ensure_ascii=False)),
        )
    if canonical is None:
        return None
    candidate_id = _candidate_id(entity_id, canonical)
    db.execute(
        "INSERT OR IGNORE INTO resolution_candidate VALUES (?, ?, ?, "
        "'same_character', 0.9, '{}', 'fixture', 'candidate', 'asset:moegirl')",
        (candidate_id, entity_id, canonical),
    )
    return candidate_id


def _write_compiled(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compiled_row(
    candidate_id: str, final_decision: str, source: str = "rule"
) -> dict:
    return {
        "candidate_id": candidate_id,
        "final_decision": final_decision,
        "decision_source": source,
        "confidence": 1.0,
        "matched_evidence": ["same_full_name"],
        "conflicting_evidence": [],
    }


def _manual_row(
    candidate_id: str, left: str, right: str, decision: str
) -> dict:
    return {
        "candidate_id": candidate_id,
        "left_entity_id": left,
        "right_entity_id": right,
        "decision": decision,
        "decided_by": "human",
        "decided_at": NOW,
    }


class MergeEntitiesTests(unittest.TestCase):
    def _scenario(self, directory: str):
        root = Path(directory)
        layout = ProjectLayout(root)
        layout.initialize()
        database = _setup_db(layout)
        compiled_path = root / "resolution-compiled.jsonl"
        manual_path = layout.manual / "entity_resolution.jsonl"
        return layout, database, compiled_path, manual_path

    def test_full_field_merge_and_golden_gate(self) -> None:
        """迷你库全字段验证：别名/事实迁移、source_entity 解析、redirect、
        resolution_decision、候选终态、entity 状态，以及黄金集门禁通过。"""
        with tempfile.TemporaryDirectory() as directory:
            layout, database, compiled_path, manual_path = self._scenario(directory)
            with closing(sqlite3.connect(database)) as db, db:
                _add_bgm_character(db, "bgm:character:1", "泉此方", ("此方",))
                candidate_id = _add_moegirl_character(
                    db,
                    "uuid5:character:aaa",
                    "泉此方",
                    aliases=("泉此方", "宅方"),
                    gender="female",
                    canonical="bgm:character:1",
                )
            _write_compiled(compiled_path, [_compiled_row(candidate_id, "auto_same")])
            manual_path.write_text(
                json.dumps(
                    _manual_row(candidate_id, "uuid5:character:aaa", "bgm:character:1", "same"),
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            plan = load_merge_map(
                database, resolution_compiled=compiled_path, manual_resolution=manual_path
            )
            self.assertEqual(plan.merge_count, 1)
            report = apply_merges(
                database,
                plan,
                build_id=BUILD_ID,
                rejected_dir=layout.rejected,
                quality_dir=layout.manifests / "quality",
            )
            self.assertEqual(report.merges_applied, 1)
            self.assertEqual(report.redirects, 1)
            self.assertEqual(report.decisions, 1)
            self.assertEqual(report.alias_conflicts, 0)
            self.assertEqual(report.fact_conflicts, 0)

            golden = verify_golden(
                database,
                manual_resolution=manual_path,
                plan=plan,
                min_canonical_with_moegirl=1,
            )
            self.assertEqual(golden.human_checked, 1)
            self.assertEqual(golden.redirect_count, 1)
            self.assertEqual(golden.merged_entity_count, 1)
            self.assertEqual(golden.canonical_with_moegirl, 1)

            with closing(sqlite3.connect(database)) as db:
                redirect = db.execute(
                    "SELECT old_entity_id, canonical_entity_id, reason "
                    "FROM entity_redirect"
                ).fetchone()
                self.assertEqual(redirect[0], "uuid5:character:aaa")
                self.assertEqual(redirect[1], "bgm:character:1")
                self.assertTrue(redirect[2].startswith("auto_same:"))

                alias_entities = {
                    row[0]
                    for row in db.execute(
                        "SELECT DISTINCT entity_id FROM entity_alias "
                        "WHERE source_kind='moegirl_dataset'"
                    )
                }
                self.assertEqual(alias_entities, {"bgm:character:1"})

                fact_entities = {
                    row[0] for row in db.execute("SELECT DISTINCT entity_id FROM entity_fact")
                }
                self.assertEqual(fact_entities, {"bgm:character:1"})

                source_entity = db.execute(
                    "SELECT entity_id, mapping_status FROM source_entity "
                    "WHERE id='uuid5:character:aaa'"
                ).fetchone()
                self.assertEqual(source_entity[0], "bgm:character:1")
                self.assertEqual(source_entity[1], "resolved")

                decision = db.execute(
                    "SELECT left_source_entity_id, right_source_entity_id, decision "
                    "FROM resolution_decision"
                ).fetchone()
                self.assertEqual(decision[0], "uuid5:character:aaa")
                self.assertEqual(decision[1], "bgm:character:1")
                self.assertEqual(decision[2], "same")

                candidate_status = db.execute(
                    "SELECT status FROM resolution_candidate WHERE id=?", (candidate_id,)
                ).fetchone()[0]
                self.assertEqual(candidate_status, "merged")

                old_status = db.execute(
                    "SELECT status FROM entity WHERE id='uuid5:character:aaa'"
                ).fetchone()[0]
                self.assertEqual(old_status, "merged")

    def test_apply_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout, database, compiled_path, manual_path = self._scenario(directory)
            with closing(sqlite3.connect(database)) as db, db:
                _add_bgm_character(db, "bgm:character:1", "泉此方")
                candidate_id = _add_moegirl_character(
                    db,
                    "uuid5:character:aaa",
                    "泉此方",
                    aliases=("宅方",),
                    gender="female",
                    canonical="bgm:character:1",
                )
            _write_compiled(compiled_path, [_compiled_row(candidate_id, "auto_same")])
            manual_path.write_text("", encoding="utf-8")

            plan = load_merge_map(
                database, resolution_compiled=compiled_path, manual_resolution=manual_path
            )
            first = apply_merges(
                database,
                plan,
                build_id=BUILD_ID,
                rejected_dir=layout.rejected,
                quality_dir=layout.manifests / "quality",
            )
            second = apply_merges(
                database,
                plan,
                build_id=BUILD_ID,
                rejected_dir=layout.rejected,
                quality_dir=layout.manifests / "quality",
            )
            self.assertEqual(first.redirects, 1)
            self.assertEqual(second.redirects, 0)  # INSERT OR IGNORE 重跑不新增
            self.assertEqual(second.decisions, 0)
            with closing(sqlite3.connect(database)) as db:
                self.assertEqual(
                    db.execute("SELECT count(*) FROM entity_redirect").fetchone()[0], 1
                )
                self.assertEqual(
                    db.execute("SELECT count(*) FROM resolution_decision").fetchone()[0], 1
                )
                self.assertEqual(
                    db.execute(
                        "SELECT count(*) FROM entity_alias WHERE entity_id='bgm:character:1' "
                        "AND source_kind='moegirl_dataset'"
                    ).fetchone()[0],
                    1,
                )

    def test_one_old_to_two_canonicals_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout, database, compiled_path, manual_path = self._scenario(directory)
            with closing(sqlite3.connect(database)) as db, db:
                _add_bgm_character(db, "bgm:character:1", "A")
                _add_bgm_character(db, "bgm:character:2", "B")
                candidate_id = _add_moegirl_character(
                    db, "uuid5:character:aaa", "A", canonical="bgm:character:1"
                )
            _write_compiled(compiled_path, [_compiled_row(candidate_id, "auto_same")])
            # 人工又把同一萌百实体判为与另一个 BGM 实体相同 → 一对多硬冲突。
            manual_path.write_text(
                json.dumps(
                    _manual_row(
                        "candidate:manual", "uuid5:character:aaa", "bgm:character:2", "same"
                    ),
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(MergeConflictError) as ctx:
                load_merge_map(
                    database,
                    resolution_compiled=compiled_path,
                    manual_resolution=manual_path,
                )
            self.assertEqual(ctx.exception.conflicts[0]["old_entity_id"], "uuid5:character:aaa")

    def test_human_veto_resolves_one_to_many(self) -> None:
        """人工 different 否决多余目标后，一对多冲突应消失（§1.2 规则2先于规则3）。"""
        with tempfile.TemporaryDirectory() as directory:
            layout, database, compiled_path, manual_path = self._scenario(directory)
            with closing(sqlite3.connect(database)) as db, db:
                _add_bgm_character(db, "bgm:character:1", "A")
                _add_bgm_character(db, "bgm:character:2", "B")
                first = _add_moegirl_character(
                    db, "uuid5:character:aaa", "A", canonical="bgm:character:1"
                )
                second = _candidate_id("uuid5:character:aaa", "bgm:character:2")
                db.execute(
                    "INSERT OR IGNORE INTO resolution_candidate VALUES (?, ?, ?, "
                    "'same_character', 0.9, '{}', 'fixture', 'candidate', 'asset:moegirl')",
                    (second, "uuid5:character:aaa", "bgm:character:2"),
                )
            _write_compiled(
                compiled_path,
                [_compiled_row(first, "auto_same"), _compiled_row(second, "auto_same")],
            )
            manual_path.write_text(
                json.dumps(
                    _manual_row(
                        first, "uuid5:character:aaa", "bgm:character:1", "different"
                    ),
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            plan = load_merge_map(
                database, resolution_compiled=compiled_path, manual_resolution=manual_path
            )
            self.assertEqual(plan.merge_count, 1)
            self.assertEqual(
                plan.merges["uuid5:character:aaa"].canonical_entity_id, "bgm:character:2"
            )

    def test_manual_supersede_latest_wins(self) -> None:
        """同一 candidate_id 追加新裁决覆盖旧裁决：先 same 后 different → 否决，门禁只验最新。"""
        with tempfile.TemporaryDirectory() as directory:
            layout, database, compiled_path, manual_path = self._scenario(directory)
            with closing(sqlite3.connect(database)) as db, db:
                _add_bgm_character(db, "bgm:character:1", "A")
                candidate_id = _add_moegirl_character(
                    db, "uuid5:character:aaa", "A", canonical="bgm:character:1"
                )
            _write_compiled(compiled_path, [_compiled_row(candidate_id, "auto_same")])
            lines = [
                _manual_row(candidate_id, "uuid5:character:aaa", "bgm:character:1", "same"),
                _manual_row(
                    candidate_id, "uuid5:character:aaa", "bgm:character:1", "different"
                ),
            ]
            manual_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in lines) + "\n",
                encoding="utf-8",
            )
            plan = load_merge_map(
                database, resolution_compiled=compiled_path, manual_resolution=manual_path
            )
            self.assertEqual(plan.merge_count, 0)  # different 覆盖了 same
            golden = verify_golden(
                database,
                manual_resolution=manual_path,
                plan=plan,
                min_canonical_with_moegirl=0,
            )
            self.assertEqual(golden.human_checked, 1)
            self.assertEqual(golden.redirect_count, 0)

    def test_human_veto_overrides_auto_same(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout, database, compiled_path, manual_path = self._scenario(directory)
            with closing(sqlite3.connect(database)) as db, db:
                _add_bgm_character(db, "bgm:character:1", "A")
                candidate_id = _add_moegirl_character(
                    db, "uuid5:character:aaa", "A", canonical="bgm:character:1"
                )
            _write_compiled(compiled_path, [_compiled_row(candidate_id, "auto_same")])
            manual_path.write_text(
                json.dumps(
                    _manual_row(
                        candidate_id, "uuid5:character:aaa", "bgm:character:1", "different"
                    ),
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            plan = load_merge_map(
                database, resolution_compiled=compiled_path, manual_resolution=manual_path
            )
            self.assertEqual(plan.merge_count, 0)
            self.assertEqual(plan.candidate_statuses[candidate_id], "rejected")
            apply_merges(
                database,
                plan,
                build_id=BUILD_ID,
                rejected_dir=layout.rejected,
                quality_dir=layout.manifests / "quality",
            )
            # 人工否决后不得存在 redirect，黄金集门禁仍通过。
            golden = verify_golden(
                database,
                manual_resolution=manual_path,
                plan=plan,
                min_canonical_with_moegirl=0,
            )
            self.assertEqual(golden.redirect_count, 0)
            self.assertEqual(golden.human_passed, 1)

    def test_redirect_chain_normalizes_to_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout, database, compiled_path, manual_path = self._scenario(directory)
            with closing(sqlite3.connect(database)) as db, db:
                _add_bgm_character(db, "bgm:character:9", "C")
                _add_bgm_character(db, "bgm:character:1", "B")
                _add_moegirl_character(db, "uuid5:character:aaa", "A")
            _write_compiled(compiled_path, [])
            # 人工裁决构成链：uuid5:A → bgm:B，bgm:B → bgm:C，须归一到 C。
            lines = [
                _manual_row("candidate:m1", "uuid5:character:aaa", "bgm:character:1", "same"),
                _manual_row("candidate:m2", "bgm:character:1", "bgm:character:9", "same"),
            ]
            manual_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in lines) + "\n",
                encoding="utf-8",
            )
            plan = load_merge_map(
                database, resolution_compiled=compiled_path, manual_resolution=manual_path
            )
            self.assertEqual(
                plan.merges["uuid5:character:aaa"].canonical_entity_id, "bgm:character:9"
            )
            self.assertEqual(
                plan.merges["bgm:character:1"].canonical_entity_id, "bgm:character:9"
            )

    def test_query_follows_redirect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout, database, compiled_path, manual_path = self._scenario(directory)
            with closing(sqlite3.connect(database)) as db, db:
                _add_bgm_character(db, "bgm:character:1", "泉此方")
                # 旧实体带一个独有的 canonical_name（不作别名，故不会迁到规范侧），
                # 以及一个会迁到规范侧的别名“宅方”。
                candidate_id = _add_moegirl_character(
                    db,
                    "uuid5:character:aaa",
                    "萌百泉此方",
                    aliases=("宅方",),
                    canonical="bgm:character:1",
                )
            _write_compiled(compiled_path, [_compiled_row(candidate_id, "auto_same")])
            manual_path.write_text("", encoding="utf-8")
            plan = load_merge_map(
                database, resolution_compiled=compiled_path, manual_resolution=manual_path
            )
            apply_merges(
                database,
                plan,
                build_id=BUILD_ID,
                rejected_dir=layout.rejected,
                quality_dir=layout.manifests / "quality",
            )
            SearchIndexer.rebuild(database)

            # 萌百别名“宅方”已迁到规范实体，直接命中 BGM 角色。
            hits = KnowledgeQueries.search_entities(database, "宅方")
            self.assertTrue(hits)
            self.assertEqual(hits[0].entity_id, "bgm:character:1")
            self.assertIsNone(hits[0].redirected_from)

            # 直查旧实体独有的名称 → 跟随 redirect 到规范实体并附 redirected_from。
            hits_by_old_name = KnowledgeQueries.search_entities(database, "萌百泉此方")
            target = next(
                hit for hit in hits_by_old_name if hit.entity_id == "bgm:character:1"
            )
            self.assertEqual(target.redirected_from, "uuid5:character:aaa")
            self.assertEqual(target.canonical_name, "泉此方")
            # 去重：同一规范实体只出现一次。
            self.assertEqual(
                sum(1 for hit in hits_by_old_name if hit.entity_id == "bgm:character:1"), 1
            )

    def test_n_to_one_fact_conflict_logged(self) -> None:
        """两个萌百实体并入同一 BGM，gender 事实共享 asset 冲突：落盘并保留其一。"""
        with tempfile.TemporaryDirectory() as directory:
            layout, database, compiled_path, manual_path = self._scenario(directory)
            with closing(sqlite3.connect(database)) as db, db:
                _add_bgm_character(db, "bgm:character:1", "A")
                first = _add_moegirl_character(
                    db,
                    "uuid5:character:aaa",
                    "A",
                    aliases=("别名甲",),
                    gender="female",
                    canonical="bgm:character:1",
                )
                second = _add_moegirl_character(
                    db,
                    "uuid5:character:bbb",
                    "A",
                    aliases=("别名乙",),
                    gender="male",
                    canonical="bgm:character:1",
                )
            _write_compiled(
                compiled_path,
                [
                    _compiled_row(first, "auto_same"),
                    _compiled_row(second, "auto_same"),
                ],
            )
            manual_path.write_text("", encoding="utf-8")
            plan = load_merge_map(
                database, resolution_compiled=compiled_path, manual_resolution=manual_path
            )
            # N:1 且全非人工 → awareness 列出供知情，不拒绝。
            self.assertEqual(len(plan.awareness), 1)
            self.assertEqual(plan.awareness[0]["canonical_entity_id"], "bgm:character:1")

            report = apply_merges(
                database,
                plan,
                build_id=BUILD_ID,
                rejected_dir=layout.rejected,
                quality_dir=layout.manifests / "quality",
            )
            self.assertEqual(report.fact_conflicts, 1)
            self.assertIsNotNone(report.fact_conflict_log)
            conflict_lines = Path(report.fact_conflict_log).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(conflict_lines), 1)
            conflict = json.loads(conflict_lines[0])
            self.assertEqual(conflict["fact_kind"], "gender")
            self.assertIn("kept", conflict)
            self.assertIn("dropped", conflict)
            with closing(sqlite3.connect(database)) as db:
                # UNIQUE(entity_id, fact_kind, source_asset_id) 下只保留一条。
                self.assertEqual(
                    db.execute(
                        "SELECT count(*) FROM entity_fact WHERE entity_id='bgm:character:1' "
                        "AND fact_kind='gender'"
                    ).fetchone()[0],
                    1,
                )
                # 两个萌百别名都迁到规范实体（source_kind 相同但 normalized 不同，无冲突）。
                self.assertEqual(
                    db.execute(
                        "SELECT count(*) FROM entity_alias WHERE entity_id='bgm:character:1' "
                        "AND source_kind='moegirl_dataset'"
                    ).fetchone()[0],
                    2,
                )

    def test_golden_gate_fails_when_different_has_redirect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout, database, compiled_path, manual_path = self._scenario(directory)
            with closing(sqlite3.connect(database)) as db, db:
                _add_bgm_character(db, "bgm:character:1", "A")
                _add_moegirl_character(db, "uuid5:character:aaa", "A")
                # 人为制造错误：人工判 different，却存在两者间的 redirect。
                db.execute(
                    "INSERT INTO entity_redirect VALUES "
                    "('uuid5:character:aaa', 'bgm:character:1', 'auto_same:rule', ?)",
                    (NOW,),
                )
            _write_compiled(compiled_path, [])
            manual_path.write_text(
                json.dumps(
                    _manual_row(
                        "candidate:m1", "uuid5:character:aaa", "bgm:character:1", "different"
                    ),
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            plan = load_merge_map(
                database, resolution_compiled=compiled_path, manual_resolution=manual_path
            )
            with self.assertRaises(GoldenGateError) as ctx:
                verify_golden(
                    database,
                    manual_resolution=manual_path,
                    plan=plan,
                    min_canonical_with_moegirl=0,
                )
            self.assertEqual(ctx.exception.failures[0]["candidate_id"], "candidate:m1")


if __name__ == "__main__":
    unittest.main()

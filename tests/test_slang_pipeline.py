from __future__ import annotations

import sqlite3
import tempfile
import unittest
import json
from contextlib import closing
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch

from mikirin.schema import CONTENT_SCHEMA
from mikirin.p1_build import stable_id
from mikirin.query import KnowledgeQueries
from mikirin.slang.pipeline import SlangPipelineConfig, discover_slang, slang_run_report
from mikirin.slang.store import AliasPolicy, apply_arbitration_results, load_arbitration_candidates


class SlangPipelineTests(unittest.TestCase):
    def _small_config(self):
        return SlangPipelineConfig(minimum_local_occurrences=1, minimum_local_documents=1,
                                   minimum_global_occurrences=2, minimum_global_documents=2,
                                   minimum_global_authors=2, candidate_score_threshold=0,
                                   single_char_score_threshold=0)

    def test_incremental_reuses_unchanged_subject_and_matches_full_run(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            config = self._small_config()
            discover_slang(database, build_id="build-p1", run_id="base", config=config)
            with patch("mikirin.slang.pipeline._scan_subject", side_effect=AssertionError("must reuse")):
                result = discover_slang(database, build_id="build-p1", run_id="incremental",
                                        config=replace(config, incremental_base_run_id="base"))
            self.assertEqual(result["subjects_reused"], 1)
            with closing(sqlite3.connect(database)) as db:
                select = ("SELECT normalized_candidate,occurrence_count,document_count,author_count,score "
                          "FROM slang_candidate WHERE run_id=? ORDER BY normalized_candidate")
                self.assertEqual(db.execute(select, ("base",)).fetchall(),
                                 db.execute(select, ("incremental",)).fetchall())
                self.assertEqual(db.execute("SELECT count(*) FROM cooccurrence WHERE run_id='base'").fetchone(),
                                 db.execute("SELECT count(*) FROM cooccurrence WHERE run_id='incremental'").fetchone())

    def test_frozen_vector_neighbors_are_linked_and_bound_to_run_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            config = replace(self._small_config(), vector_neighbors_by_candidate={
                "锯": [{"entity_id": "character:denji", "score": .99}],
            })
            discover_slang(database, build_id="build-p1", run_id="vectors", config=config)
            with closing(sqlite3.connect(database)) as db:
                rows = db.execute("SELECT l.entity_id,l.evidence_json FROM slang_link_candidate l "
                                  "JOIN slang_candidate c ON c.id=l.candidate_id "
                                  "WHERE c.run_id='vectors' AND c.normalized_candidate='锯' ORDER BY l.rank").fetchall()
            self.assertEqual(rows[0][0], "character:denji")
            self.assertEqual(json.loads(rows[0][1])["retrieval"], "external_top3")
            self.assertTrue(any("official_name_exclusion" in json.loads(row[1]) for row in rows))
            with self.assertRaisesRegex(ValueError, "config mismatch"):
                discover_slang(database, build_id="build-p1", run_id="vectors", config=self._small_config())

    def test_crash_after_subject_checkpoint_resumes_without_rescan(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            config = self._small_config()
            with patch("mikirin.slang.pipeline._aggregate_candidates", side_effect=RuntimeError("crash")):
                with self.assertRaisesRegex(RuntimeError, "crash"):
                    discover_slang(database, build_id="build-p1", run_id="crash", config=config)
            with patch("mikirin.slang.pipeline._scan_subject", side_effect=AssertionError("already checkpointed")):
                result = discover_slang(database, build_id="build-p1", run_id="crash", config=config)
            self.assertGreater(result["candidates"], 0)
            self.assertEqual(slang_run_report(database, run_id="crash")["status"], "completed")

    def test_incremental_changed_documents_and_aliases_are_not_reused(self):
        from mikirin.slang.pipeline import _scan_subject
        for mutate in ("UPDATE source_document SET content=content || '新评论' WHERE id='document:1'",
                       "UPDATE entity_alias SET alias_text='新名',normalized_alias='新名' WHERE id='alias:denji'"):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                database = self._database(directory)
                config = self._small_config()
                discover_slang(database, build_id="build-p1", run_id="base", config=config)
                with closing(sqlite3.connect(database)) as db, db:
                    db.execute(mutate)
                with patch("mikirin.slang.pipeline._scan_subject", wraps=_scan_subject) as scanner:
                    discover_slang(database, build_id="build-p1", run_id="next",
                                   config=replace(config, incremental_base_run_id="base"))
                    self.assertEqual(scanner.call_count, 1)

    def test_author_breadth_is_distinct_and_not_truncated_by_sample_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            config = replace(self._small_config(), author_sample_limit=1)
            discover_slang(database, build_id="build-p1", run_id="authors", config=config)
            with closing(sqlite3.connect(database)) as db:
                count = db.execute("SELECT author_count FROM slang_candidate WHERE run_id='authors' "
                                   "AND normalized_candidate='锯'").fetchone()[0]
                self.assertEqual(count, 4)

    def test_resume_rejects_changed_alias_dictionary(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            config = self._small_config()
            discover_slang(database, build_id="build-p1", run_id="base", config=config)
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("UPDATE entity_alias SET normalized_alias='新名' WHERE id='alias:denji'")
            with self.assertRaisesRegex(ValueError, "source watermark/content identity"):
                discover_slang(database, build_id="build-p1", run_id="base", config=config)

    def test_max_subjects_is_a_stable_scope_across_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("INSERT INTO entity VALUES('subject:z','subject','另一作品','active')")
                db.execute("INSERT INTO subjects(entity_id,type,name,tags_json,source_asset_id) "
                           "VALUES('subject:z',2,'另一作品','[]','asset')")
                db.execute("INSERT INTO source_document SELECT 'document:z',source,'record:z',title,url,"
                           "content,structured_json,content_hash,fetched_at,source_asset_id "
                           "FROM source_document WHERE id='document:1'")
                db.execute("INSERT INTO community_document SELECT 'document:z',source_kind,'subject:z',"
                           "episode_entity_id,author_hash,rating,posted_at_raw,license_id,redistributable,"
                           "attribution_required,commercial_use_allowed FROM community_document WHERE document_id='document:1'")
            config = replace(self._small_config(), max_subjects=1)
            discover_slang(database, build_id="build-p1", run_id="limited", config=config)
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("UPDATE slang_run SET status='building' WHERE id='limited'")
            with patch("mikirin.slang.pipeline._scan_subject", side_effect=AssertionError("scope drift")):
                discover_slang(database, build_id="build-p1", run_id="limited", config=config)
            self.assertEqual(slang_run_report(database, run_id="limited")["subjects_completed"], 1)

    def test_closed_store_rejects_unknown_decisions_and_nonfinite_confidence(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            discover_slang(database, build_id="build-p1", run_id="base", config=self._small_config())
            with closing(sqlite3.connect(database)) as db:
                candidate_id = db.execute("SELECT id FROM slang_candidate WHERE run_id='base' "
                                          "AND normalized_candidate='锯'").fetchone()[0]
            row = {"candidate_id": candidate_id, "decision": "outside", "confidence": 1,
                   "comment_id": "document:1", "quote": "锯", "provenance": {}}
            with self.assertRaisesRegex(ValueError, "closed candidate set"):
                apply_arbitration_results(database, run_id="base", decisions=[row])
            for value in (float("nan"), float("inf"), True, -1, 2):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    apply_arbitration_results(database, run_id="base", decisions=[{**row, "confidence": value}])

    def test_default_alias_policy_requires_strictly_more_than_twenty_authors(self):
        from mikirin.slang.store import _decision_status
        for bucket in ("self_reference", "single_char"):
            self.assertEqual(_decision_status(bucket=bucket, link_score=1, confidence=1,
                                              author_count=20, policy=AliasPolicy()), "provisional")
            self.assertEqual(_decision_status(bucket=bucket, link_score=1, confidence=1,
                                              author_count=21, policy=AliasPolicy()), "confirmed")
        self.assertIsNone(_decision_status(bucket="single_char", link_score=.6, confidence=.8,
                                           author_count=100, policy=AliasPolicy()))

    def _database(self, directory: str) -> Path:
        path = Path(directory) / "content.db"
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript(CONTENT_SCHEMA)
            db.execute("INSERT INTO schema_metadata VALUES('schema_version','10')")
            db.execute(
                "INSERT INTO build(id,started_at,code_version,status) "
                "VALUES('build-p1','now','test','building')"
            )
            db.execute(
                "INSERT INTO ingestion_run(id,source,started_at,source_version,parser_version,status) "
                "VALUES('run','test','now','v','v','done')"
            )
            db.execute(
                "INSERT INTO source_asset(id,ingestion_run_id,local_path,content_hash,fetched_at) "
                "VALUES('asset','run','x','hash','now')"
            )
            db.executemany(
                "INSERT INTO entity VALUES(?,?,?,'active')",
                [
                    ("subject:chainsaw", "subject", "电锯人"),
                    ("character:denji", "character", "电次"),
                ],
            )
            db.execute(
                "INSERT INTO subjects(entity_id,type,name,tags_json,source_asset_id) "
                "VALUES('subject:chainsaw',2,'电锯人','[]','asset')"
            )
            db.execute(
                "INSERT INTO characters(entity_id,name,source_asset_id) "
                "VALUES('character:denji','电次','asset')"
            )
            db.execute(
                "INSERT INTO rel_subject_character VALUES('subject:chainsaw','character:denji',1,1,'asset')"
            )
            db.executemany(
                "INSERT INTO entity_alias VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        "alias:chainsaw",
                        "subject:chainsaw",
                        "电锯人",
                        "电锯人",
                        "zh",
                        "bgm_archive",
                        "active",
                        1.0,
                        "asset",
                        "build-p1",
                        "unknown",
                    ),
                    (
                        "alias:denji",
                        "character:denji",
                        "电次",
                        "电次",
                        "zh",
                        "bgm_archive",
                        "active",
                        1.0,
                        "asset",
                        "build-p1",
                        "unknown",
                    ),
                ],
            )
            comments = [
                "大家都叫它锯，锯这次真痛快。电锯人和电次都很亮眼。",
                "我追锯就是为了这种疯劲，锯的节奏很好。",
                "锯今天又发力了，大家说锯都知道是哪部。",
                "看锯的人都在讨论这一幕，锯确实有辨识度。",
            ]
            for index, content in enumerate(comments, 1):
                document_id = f"document:{index}"
                db.execute(
                    "INSERT INTO source_document VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        document_id,
                        "test",
                        str(index),
                        "comment",
                        None,
                        content,
                        "{}",
                        f"hash:{index}",
                        None,
                        "asset",
                    ),
                )
                db.execute(
                    "INSERT INTO community_document VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        document_id,
                        "bgm_subject_comment",
                        "subject:chainsaw",
                        None,
                        f"author:{index}",
                        None,
                        "now",
                        "unknown",
                        1,
                        0,
                        1,
                    ),
                )
        return path

    def test_full_scan_is_resumable_and_writes_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            config = SlangPipelineConfig(
                minimum_local_occurrences=1,
                minimum_local_documents=1,
                minimum_global_occurrences=2,
                minimum_global_documents=2,
                minimum_global_authors=2,
                candidate_score_threshold=0.0,
                single_char_score_threshold=0.0,
                candidate_limit=100,
                local_candidate_limit=1_000,
            )
            first = discover_slang(
                database,
                build_id="build-p1",
                run_id="slang-test",
                config=config,
            )
            second = discover_slang(
                database,
                build_id="build-p1",
                run_id="slang-test",
                config=config,
            )
            report = slang_run_report(database, run_id="slang-test")

            self.assertEqual(report["status"], "completed")
            self.assertEqual(first["run_id"], second["run_id"])
            self.assertEqual(report["subjects_completed"], 1)
            self.assertGreater(report["cooccurrences"], 0)
            with closing(sqlite3.connect(database)) as db:
                row = db.execute(
                    "SELECT id,bucket FROM slang_candidate "
                    "WHERE run_id='slang-test' AND normalized_candidate='锯'"
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[1], "single_char")
                evidence = db.execute(
                    "SELECT context_text FROM slang_evidence WHERE candidate_id=?", (row[0],)
                ).fetchall()
                self.assertTrue(evidence)
                self.assertTrue(all("锯" in value[0] for value in evidence))

                link = db.execute(
                    "SELECT entity_id FROM slang_link_candidate WHERE candidate_id=? "
                    "ORDER BY rank LIMIT 1",
                    (row[0],),
                ).fetchone()[0]
            arbitration_inputs = load_arbitration_candidates(
                database, run_id="slang-test", limit=100
            )
            self.assertTrue(any(item.candidate_id == row[0] for item in arbitration_inputs))
            applied = apply_arbitration_results(
                database,
                run_id="slang-test",
                decisions=[
                    {
                        "candidate_id": row[0],
                        "decision": link,
                        "confidence": 1.0,
                        "comment_id": "document:1",
                        "quote": "锯",
                        "reason": "test",
                        "provenance": {"cache_key": "test"},
                    }
                ],
                policy=AliasPolicy(
                    confirmed_confidence=0.0,
                    provisional_confidence=0.0,
                    confirmed_link_score=0.0,
                    provisional_link_score=0.0,
                    single_confirmed_confidence=0.0,
                    single_confirmed_link_score=0.0,
                    confirmed_author_breadth=0,
                ),
            )
            self.assertEqual(applied["confirmed"], 1)
            hits = KnowledgeQueries.search_entities(database, "锯", limit=3)
            self.assertTrue(hits)
            self.assertEqual(hits[0].entity_id, link)

    def test_resume_rejects_same_length_source_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            config = SlangPipelineConfig(
                minimum_local_occurrences=1,
                minimum_local_documents=1,
                minimum_global_occurrences=2,
                minimum_global_documents=2,
                minimum_global_authors=2,
                candidate_score_threshold=0.0,
                single_char_score_threshold=0.0,
            )
            discover_slang(database, build_id="build-p1", run_id="slang-resume", config=config)
            with closing(sqlite3.connect(database)) as db, db:
                content = str(
                    db.execute(
                        "SELECT content FROM source_document WHERE id='document:1'"
                    ).fetchone()[0]
                )
                changed = ("X" if content[0] != "X" else "Y") + content[1:]
                self.assertEqual(len(changed), len(content))
                db.execute(
                    "UPDATE slang_run SET status='building',completed_at=NULL WHERE id=?",
                    ("slang-resume",),
                )
                db.execute(
                    "UPDATE source_document SET content=? WHERE id='document:1'", (changed,)
                )
            with self.assertRaisesRegex(ValueError, "source watermark/content identity"):
                discover_slang(database, build_id="build-p1", run_id="slang-resume", config=config)

    def test_legacy_wrapped_watermark_is_upgraded_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            config = SlangPipelineConfig(
                minimum_local_occurrences=1,
                minimum_local_documents=1,
                minimum_global_occurrences=2,
                minimum_global_documents=2,
                minimum_global_authors=2,
                candidate_score_threshold=0.0,
                single_char_score_threshold=0.0,
            )
            discover_slang(database, build_id="build-p1", run_id="slang-legacy", config=config)
            with closing(sqlite3.connect(database)) as db, db:
                stored_watermark, stored_config_json = db.execute(
                    "SELECT source_watermark,config_json FROM slang_run WHERE id=?",
                    ("slang-legacy",),
                ).fetchone()
                current_source = json.loads(stored_watermark)
                current_source = current_source.get("source", current_source)
                legacy_source = {
                    key: current_source[key]
                    for key in ("document_count", "max_rowid", "character_count")
                }
                legacy_watermark = json.dumps(
                    legacy_source, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                legacy_config = json.loads(stored_config_json)
                legacy_config.pop("maximum_alias_count")
                legacy_config.pop("maximum_normalized_alias_characters")
                db.execute(
                    """
                    UPDATE slang_run
                    SET status='building',completed_at=NULL,source_watermark=?,
                        input_digest=?,config_json=?
                    WHERE id=?
                    """,
                    (
                        json.dumps(
                            {
                                "source": legacy_source,
                                "last_subject_entity_id": "subject:chainsaw",
                                "subjects_completed": 1,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        stable_id("slang-input", legacy_watermark, legacy_config),
                        json.dumps(legacy_config, ensure_ascii=False, sort_keys=True),
                        "slang-legacy",
                    ),
                )
            discover_slang(database, build_id="build-p1", run_id="slang-legacy", config=config)
            # A migration must be durable: the immediately following resume
            # validates the upgraded config and digest instead of failing on
            # the two newly introduced capacity fields.
            resumed_again = discover_slang(
                database,
                build_id="build-p1",
                run_id="slang-legacy",
                config=config,
            )
            self.assertEqual(resumed_again["status"], "completed")
            with closing(sqlite3.connect(database)) as db:
                upgraded_watermark, upgraded_input_digest, upgraded_config_json = db.execute(
                    "SELECT source_watermark,input_digest,config_json FROM slang_run WHERE id=?",
                    ("slang-legacy",),
                ).fetchone()
            self.assertEqual(json.loads(upgraded_watermark), current_source)
            self.assertIn("content_digest", json.loads(upgraded_watermark))
            expected_config = {
                name: getattr(config, name) for name in config.__dataclass_fields__
            }
            self.assertEqual(
                upgraded_input_digest,
                stable_id("slang-input", upgraded_watermark, expected_config),
            )
            self.assertEqual(json.loads(upgraded_config_json), expected_config)

    def test_resume_rejects_run_identity_mismatches(self) -> None:
        cases = (
            ("build_id", "build_id mismatch", {"build_id": "other-build"}),
            ("processor_version", "processor_version mismatch", {"processor_version": "old"}),
            ("threshold_version", "threshold_version mismatch", {"threshold_version": "other"}),
            ("config_json", "config mismatch", {"candidate_limit": 101}),
            ("input_digest", "input/config digest mismatch", {"input_digest": "tampered"}),
        )
        for column, message, change in cases:
            with self.subTest(column=column), tempfile.TemporaryDirectory() as directory:
                database = self._database(directory)
                config = SlangPipelineConfig(
                    minimum_local_occurrences=1,
                    minimum_local_documents=1,
                    minimum_global_occurrences=2,
                    minimum_global_documents=2,
                    minimum_global_authors=2,
                    candidate_score_threshold=0.0,
                    single_char_score_threshold=0.0,
                )
                discover_slang(
                    database, build_id="build-p1", run_id="slang-identity", config=config
                )
                with closing(sqlite3.connect(database)) as db, db:
                    if column == "config_json":
                        altered = dict(config.__dict__) if hasattr(config, "__dict__") else {
                            name: getattr(config, name) for name in config.__dataclass_fields__
                        }
                        altered["candidate_limit"] = change["candidate_limit"]
                        db.execute(
                            "UPDATE slang_run SET config_json=? WHERE id=?",
                            (json.dumps(altered, ensure_ascii=False, sort_keys=True), "slang-identity"),
                        )
                    elif column != "threshold_version":
                        db.execute(
                            f"UPDATE slang_run SET {column}=? WHERE id=?",
                            (next(iter(change.values())), "slang-identity"),
                        )
                with self.assertRaisesRegex(ValueError, message):
                    discover_slang(
                        database,
                        build_id="build-p1",
                        run_id="slang-identity",
                        config=(
                            SlangPipelineConfig(**{
                                **{
                                    name: getattr(config, name)
                                    for name in config.__dataclass_fields__
                                },
                                "threshold_version": "other",
                            })
                            if column == "threshold_version"
                            else config
                        ),
                    )

    def test_alias_count_hard_limit_is_checked_before_ac_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            with self.assertRaisesRegex(ValueError, r"alias_count=2 \(limit=1\)"):
                discover_slang(
                    database,
                    build_id="build-p1",
                    run_id="slang-alias-limit",
                    config=SlangPipelineConfig(maximum_alias_count=1),
                )

    def test_alias_normalized_character_hard_limit_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            with self.assertRaisesRegex(ValueError, "normalized_alias_characters="):
                discover_slang(
                    database,
                    build_id="build-p1",
                    run_id="slang-alias-char-limit",
                    config=SlangPipelineConfig(
                        maximum_alias_count=100,
                        maximum_normalized_alias_characters=1,
                    ),
                )


if __name__ == "__main__":
    unittest.main()

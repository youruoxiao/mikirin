from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from mikirin.graph.store import apply_implicit_results, load_implicit_candidates
from mikirin.p1_build import stable_id
from mikirin.schema import CONTENT_SCHEMA


class GraphStoreTests(unittest.TestCase):
    def test_pass_a_load_and_pass_b_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "content.db"
            with closing(sqlite3.connect(database)) as db, db:
                db.executescript(CONTENT_SCHEMA)
                db.execute("INSERT INTO schema_metadata VALUES('schema_version','10')")
                db.execute(
                    "INSERT INTO build(id,started_at,code_version,status) "
                    "VALUES('build-p1','now','test','building')"
                )
                db.execute(
                    "INSERT INTO ingestion_run(id,source,started_at,source_version,parser_version,status) "
                    "VALUES('ingest','test','now','v','v','done')"
                )
                db.execute(
                    "INSERT INTO source_asset(id,ingestion_run_id,local_path,content_hash,fetched_at) "
                    "VALUES('asset','ingest','x','hash','now')"
                )
                for entity_id, name in (("subject:a", "作品A"), ("subject:b", "作品B")):
                    db.execute("INSERT INTO entity VALUES(?,?,?,'active')", (entity_id, "subject", name))
                    db.execute(
                        "INSERT INTO subjects(entity_id,type,name,summary,source_asset_id) "
                        "VALUES(?,2,?,?,'asset')",
                        (entity_id, name, f"{name}简介"),
                    )
                db.execute(
                    "INSERT INTO source_document VALUES(?,?,?,?,?,?,?,?,?,?)",
                    ("doc:1", "test", "1", "comment", None, "作品A和作品B是朋友。", "{}", "h", None, "asset"),
                )
                db.execute(
                    "INSERT INTO community_document VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    ("doc:1", "bgm_subject_comment", "subject:a", None, "author:1", None, "now", "unknown", 1, 0, 1),
                )
                db.execute(
                    "INSERT INTO slang_run VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    ("slang-run", "build-p1", "v", "t", "w", "in", None, "{}", "now", "now", "completed"),
                )
                candidate_id = stable_id("cooccurrence", "slang-run", "doc:1", "subject:a", "subject:b")
                db.execute(
                    "INSERT INTO cooccurrence VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (candidate_id, "slang-run", "doc:1", "subject:a", "subject:a", "subject:b", "author:1", "[]", "[]", "作品A和作品B是朋友。"),
                )

            candidates = load_implicit_candidates(
                database,
                run_id="slang-run",
                pair_limit=10,
                evidence_per_pair=2,
                minimum_author_weight=1,
            )
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].candidate_id, candidate_id)
            # Database ingestion is an independent closed-set gate: it must
            # not trust callers to have passed the model response validator.
            for change in ({"relation": "invented"}, {"sentiment": "invented"},
                           {"confidence": float("nan")}, {"confidence": True}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    apply_implicit_results(database, run_id="slang-run", decisions=[{
                        "relation": "friend", "sentiment": "positive", "confidence": .9,
                        "provenance": {"candidate_id": candidate_id}, **change,
                    }])
            report = apply_implicit_results(
                database,
                run_id="slang-run",
                decisions=[
                    {
                        "src": "subject:a",
                        "dst": "subject:b",
                        "relation": "friend",
                        "sentiment": "positive",
                        "confidence": 0.9,
                        "claim_quote": "作品A和作品B是朋友",
                        "provenance": {
                            "candidate_id": candidate_id,
                            "model": "fake",
                            "cache_key": "cache",
                            "span": {"start": 0, "end": 10},
                        },
                    }
                ],
            )
            self.assertEqual(report["implicit_edges"], 1)
            with closing(sqlite3.connect(database)) as db:
                edge = db.execute(
                    "SELECT relation,weight,evidence_count,provenance_json FROM implicit_edge"
                ).fetchone()
            self.assertEqual(edge[:3], ("friend", 1, 1))
            self.assertIn("1 位不同作者", edge[3])


if __name__ == "__main__":
    unittest.main()

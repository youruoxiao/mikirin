from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from mikirin.p1_build import (
    P1BuildConfig,
    deterministic_edge_digest,
    ensure_p1_schema,
    materialize_derived_edges,
    materialize_task_pagerank,
)
from mikirin.schema import CONTENT_SCHEMA
from mikirin.graph.search import OfflineGraphSearch
from mikirin.graph.tasks import load_credit_graph
from mikirin.graph.algorithms import GraphEdge, pagerank_scores


class P1BuildTests(unittest.TestCase):
    def _task_rows(self, database, task=None):
        with closing(sqlite3.connect(database)) as db:
            return db.execute(
                "SELECT seed_entity_id,node_entity_id,graph_kind,rank,score,evidence_json,build_id "
                "FROM task_pagerank WHERE (? IS NULL OR graph_kind=?) "
                "ORDER BY graph_kind,seed_entity_id,rank,node_entity_id", (task, task),
            ).fetchall()

    def test_person_and_character_voice_seeds_use_real_relations(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            # A role with no voice credit still belongs to a work through the
            # independent subject-character relation (third input table).
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("INSERT INTO entity VALUES('character:unvoiced','character','未配音角色','active')")
                db.execute("INSERT INTO characters(entity_id,name,source_asset_id) VALUES('character:unvoiced','未配音角色','asset')")
                db.execute("INSERT INTO rel_subject_character VALUES('subject:a','character:unvoiced',1,1,'asset')")
            counts = materialize_task_pagerank(database, build_id="base")
            self.assertGreater(counts["voice_cast"], 0)
            searcher = OfflineGraphSearch(database)
            hits = searcher.search("person:cv1", mode="graph", task="voice_cast", build_id="base")
            by_id = {hit.entity_id: hit for hit in hits}
            self.assertIn("character:person:cv1:1", by_id)
            self.assertIn("subject:a", by_id)
            self.assertIn("subject:b", by_id)
            self.assertEqual(by_id["character:person:cv1:1"].kind, "character")
            self.assertEqual(by_id["subject:a"].kind, "subject")
            evidence = by_id["subject:a"].provenance[0]["evidence"]
            self.assertEqual(evidence["seed_entity_type"], "person")
            self.assertEqual(evidence["path"][0]["source_records"][0]["source_table"], "rel_person_character")
            self.assertEqual(evidence["path"][0]["source_records"][0]["source_asset_id"], "asset")
            hits = searcher.search("character:person:cv1:1", mode="graph", task="voice_cast", build_id="base")
            self.assertTrue({"person:cv1", "subject:a"}.issubset({hit.entity_id for hit in hits}))
            hits = searcher.search("character:unvoiced", mode="graph", task="voice_cast", build_id="base")
            self.assertIn("subject:a", {hit.entity_id for hit in hits})

    def test_staff_filmography_and_task_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            materialize_derived_edges(database, build_id="base")
            materialize_task_pagerank(database, build_id="base")
            searcher = OfflineGraphSearch(database)
            hits = searcher.search("person:staff1", mode="graph", task="staff_filmography", build_id="base")
            ids = {hit.entity_id for hit in hits}
            self.assertTrue({"subject:a", "subject:b"}.issubset(ids))
            self.assertFalse(any(value.startswith("character:") or value.startswith("person:cv") for value in ids))
            work = next(hit for hit in hits if hit.entity_id == "subject:a")
            record = work.provenance[0]["evidence"]["path"][0]["source_records"][0]
            self.assertEqual(record["source_table"], "rel_subject_person")
            self.assertEqual(record["position"], "staff")
            for seed, task in (("person:staff1", "voice_cast"), ("person:cv1", "staff_filmography"),
                               ("character:person:cv1:1", "staff_filmography"), ("person:cv1", "similar_work")):
                self.assertEqual(searcher.search(seed, mode="graph", task=task, build_id="base"), [])
            # Shared subjects do not connect the separate task graphs.
            hits = searcher.search("subject:a", mode="graph", task="voice_cast", build_id="base")
            self.assertFalse(any(hit.entity_id.startswith("person:staff") for hit in hits))

    def test_credit_graph_rebuild_is_deterministic_and_preserves_similar_work(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            materialize_derived_edges(database, build_id="base")
            edge_digest = deterministic_edge_digest(database, build_id="base")
            materialize_task_pagerank(database, build_id="base")
            first = self._task_rows(database)
            similar = self._task_rows(database, "similar_work")
            materialize_task_pagerank(database, build_id="base")
            self.assertEqual(first, self._task_rows(database))
            with closing(sqlite3.connect(database)) as db, db:
                db.execute("DELETE FROM rel_person_character")
                db.execute("DELETE FROM rel_subject_person")
            materialize_task_pagerank(database, build_id="base")
            self.assertEqual(similar, self._task_rows(database, "similar_work"))
            self.assertEqual(edge_digest, deterministic_edge_digest(database, build_id="base"))
            self.assertEqual(self._task_rows(database, "voice_cast"), [])
            self.assertEqual(self._task_rows(database, "staff_filmography"), [])

    def test_similar_work_keeps_original_weighted_projection_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            with closing(sqlite3.connect(database)) as db, db:
                db.executemany("INSERT INTO derived_edge VALUES(?,?,?,?,?,?)", [
                    ("subject:a", "subject:b", "tag_similarity", 2, "{}", "base"),
                    ("subject:b", "subject:c", "same_season", 1, "{}", "base"),
                    # This unrelated projection must not enter similar_work.
                    ("subject:a", "subject:d", "staff_overlap", 100, "{}", "base"),
                ])
            materialize_task_pagerank(database, build_id="base", top_k=2, maximum_task_nodes=16)
            expected = pagerank_scores([
                GraphEdge("subject:a", "subject:b", 2), GraphEdge("subject:b", "subject:c", 1),
            ], {"subject:a": 1}, task_nodes={"subject:a", "subject:b", "subject:c"},
                restart=.15, max_iterations=200, tolerance=1e-10)
            rows = [row for row in self._task_rows(database, "similar_work") if row[0] == "subject:a"]
            self.assertEqual({row[1] for row in rows}, {"subject:b", "subject:c"})
            for row in rows:
                self.assertAlmostEqual(row[4], expected[row[1]], places=14)

    def test_duplicate_assets_do_not_multiply_credit_edge_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            with closing(sqlite3.connect(database)) as db, db:
                first = {task: load_credit_graph(db, task).edges for task in ("voice_cast", "staff_filmography")}
                db.execute("INSERT INTO source_asset(id,ingestion_run_id,local_path,content_hash,fetched_at) "
                           "VALUES('asset2','run','y','hash2','now')")
                db.execute("INSERT INTO rel_person_character SELECT person_id,subject_id,character_id,summary,'asset2' FROM rel_person_character")
                db.execute("INSERT INTO rel_subject_person SELECT subject_id,person_id,position,appear_eps_json,'asset2' FROM rel_subject_person")
                for task, edges in first.items():
                    self.assertEqual(edges, load_credit_graph(db, task).edges)

    def test_credit_search_is_lookup_only_and_validates_materialized_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            materialize_task_pagerank(database, build_id="base", restart=.2, top_k=4)
            with closing(sqlite3.connect(database)) as db:
                def authorize(action, table, column, db_name, origin):
                    if action == sqlite3.SQLITE_READ and table in {
                        "rel_person_character", "rel_subject_character", "rel_subject_person",
                        "derived_edge", "node_pagerank",
                    }:
                        return sqlite3.SQLITE_DENY
                    return sqlite3.SQLITE_OK
                db.set_authorizer(authorize)
                searcher = OfflineGraphSearch(db)
                with patch("mikirin.graph.algorithms.pagerank_scores", side_effect=AssertionError("online PPR")):
                    hits = searcher.search("person:cv1", mode="graph", task="voice_cast",
                                           build_id="base", restart=.2, top_k=4)
                    self.assertEqual(len(hits), 4)
                    with self.assertRaisesRegex(ValueError, "restart"):
                        searcher.search("person:cv1", mode="graph", task="voice_cast", build_id="base", top_k=4)
                    with self.assertRaisesRegex(ValueError, "top_k"):
                        searcher.search("person:cv1", mode="graph", task="voice_cast", build_id="base", restart=.2)
                    with self.assertRaisesRegex(ValueError, "edge kinds"):
                        searcher.search("person:cv1", mode="graph", task="voice_cast", build_id="base",
                                        restart=.2, top_k=4, edge_kinds=("cv_overlap",))
                    self.assertEqual(searcher.search("person:cv1", mode="graph", task="voice_cast",
                                                     build_id="other", restart=.2, top_k=4), [])

    def test_credit_build_failure_rolls_back_previous_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            materialize_task_pagerank(database, build_id="base")
            before = self._task_rows(database)
            with patch("mikirin.graph.tasks.load_credit_graph", side_effect=RuntimeError("failed")):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    materialize_task_pagerank(database, build_id="base")
            self.assertEqual(before, self._task_rows(database))

    def _database(self, directory: str) -> Path:
        path = Path(directory) / "content.db"
        with closing(sqlite3.connect(path)) as db, db:
            db.executescript(CONTENT_SCHEMA)
            db.execute("INSERT INTO schema_metadata VALUES('schema_version','10')")
            db.execute(
                "INSERT INTO build(id,started_at,code_version,status) "
                "VALUES('base','now','test','published')"
            )
            db.execute(
                "INSERT INTO ingestion_run(id,source,started_at,source_version,parser_version,status) "
                "VALUES('run','test','now','v','v','done')"
            )
            db.execute(
                "INSERT INTO source_asset(id,ingestion_run_id,local_path,content_hash,fetched_at) "
                "VALUES('asset','run','x','hash','now')"
            )
            subjects = [
                ("subject:a", "A", "2024-01-01", ["rare", "common"]),
                ("subject:b", "B", "2024-02-01", ["rare", "common"]),
                ("subject:c", "C", "2024-03-01", ["other", "common"]),
                ("subject:d", "D", "2024-07-01", ["other"]),
            ]
            for index, (entity_id, name, date, tags) in enumerate(subjects, 1):
                db.execute(
                    "INSERT INTO entity VALUES(?,?,?,'active')",
                    (entity_id, "subject", name),
                )
                db.execute(
                    "INSERT INTO subjects(entity_id,type,name,date,tags_json,source_asset_id) "
                    "VALUES(?,2,?,?,?,'asset')",
                    (
                        entity_id,
                        name,
                        date,
                        json.dumps(
                            [{"name": tag, "count": 1} for tag in tags],
                            separators=(",", ":"),
                        ),
                    ),
                )
                db.execute(
                    "INSERT INTO score_snapshot(subject_entity_id,build_id,rank) VALUES(?,?,?)",
                    (entity_id, "base", index),
                )
            for person_id in ("person:staff1", "person:staff2", "person:cv1", "person:cv2"):
                db.execute("INSERT INTO entity VALUES(?,?,?,'active')", (person_id, "person", person_id))
                db.execute(
                    "INSERT INTO persons(entity_id,name,source_asset_id) VALUES(?,?,'asset')",
                    (person_id, person_id),
                )
            for person_id in ("person:staff1", "person:staff2"):
                for subject_id in ("subject:a", "subject:b"):
                    db.execute(
                        "INSERT INTO rel_subject_person VALUES(?,?,?,'[]','asset')",
                        (subject_id, person_id, "staff"),
                    )
            for person_id in ("person:cv1", "person:cv2"):
                for offset, subject_id in enumerate(("subject:a", "subject:b"), 1):
                    character_id = f"character:{person_id}:{offset}"
                    db.execute(
                        "INSERT INTO entity VALUES(?,?,?,'active')",
                        (character_id, "character", character_id),
                    )
                    db.execute(
                        "INSERT INTO characters(entity_id,name,source_asset_id) VALUES(?,?,'asset')",
                        (character_id, character_id),
                    )
                    db.execute(
                        "INSERT INTO rel_person_character VALUES(?,?,?,?,?)",
                        (person_id, subject_id, character_id, "", "asset"),
                    )
        return path

    def test_schema_upgrade_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            with closing(sqlite3.connect(database)) as db:
                ensure_p1_schema(db)
                ensure_p1_schema(db)
                version = db.execute(
                    "SELECT value FROM schema_metadata WHERE key='schema_version'"
                ).fetchone()[0]
            self.assertEqual(version, "10")

    def test_derived_edges_are_deterministic_and_idf_weighted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = self._database(directory)
            config = P1BuildConfig(
                staff_degree_cap=10,
                cv_degree_cap=10,
                tag_df_cap=10,
                tag_min_similarity=0.05,
                same_season_neighbors=2,
                edge_top_k=4,
            )
            first = materialize_derived_edges(database, build_id="base", config=config)
            first_digest = deterministic_edge_digest(database, build_id="base")
            second = materialize_derived_edges(database, build_id="base", config=config)
            second_digest = deterministic_edge_digest(database, build_id="base")

            self.assertEqual(first, second)
            self.assertEqual(first_digest, second_digest)
            self.assertGreaterEqual(first["staff_overlap"], 1)
            self.assertGreaterEqual(first["cv_overlap"], 1)
            self.assertGreaterEqual(first["tag_similarity"], 1)
            self.assertGreaterEqual(first["same_season"], 1)
            with closing(sqlite3.connect(database)) as db:
                weight = db.execute(
                    "SELECT weight FROM derived_edge WHERE kind='tag_similarity' "
                    "AND src_entity_id='subject:a' AND dst_entity_id='subject:b'"
                ).fetchone()[0]
            self.assertGreater(weight, 0.05)
            self.assertLessEqual(weight, 1.0)

            task_counts = materialize_task_pagerank(
                database,
                build_id="base",
                top_k=2,
                maximum_task_nodes=16,
            )
            self.assertGreater(task_counts["similar_work"], 0)
            with closing(sqlite3.connect(database)) as db:
                task_row = db.execute(
                    "SELECT seed_entity_id,node_entity_id,rank FROM task_pagerank "
                    "WHERE graph_kind='similar_work' ORDER BY seed_entity_id,rank LIMIT 1"
                ).fetchone()
            self.assertNotEqual(task_row[0], task_row[1])
            self.assertGreaterEqual(task_row[2], 1)


if __name__ == "__main__":
    unittest.main()

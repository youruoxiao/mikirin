import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from mikirin.graph.search import (
    DEFAULT_TOP_K,
    GraphTask,
    OfflineGraphSearch,
    build_ab_package,
)


class OfflineGraphSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.database = Path(self.tmp.name) / "minimal.db"
        with closing(sqlite3.connect(self.database)) as db, db:
            db.executescript(
                """
                CREATE TABLE derived_edge (
                    src_entity_id TEXT NOT NULL,
                    dst_entity_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    weight REAL NOT NULL,
                    evidence_json TEXT NOT NULL,
                    build_id TEXT NOT NULL
                );
                CREATE TABLE node_pagerank (
                    node_entity_id TEXT NOT NULL,
                    graph_kind TEXT NOT NULL,
                    score REAL NOT NULL,
                    build_id TEXT NOT NULL
                );
                """
            )
            db.executemany(
                "INSERT INTO derived_edge VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ("seed", "work-b", "tag_similarity", 0.90, '{"tag":"x"}', "b1"),
                    ("seed", "work-c", "tag_similarity", 0.80, "{}", "b1"),
                    ("seed", "wrong-kind", "staff_overlap", 0.99, "{}", "b1"),
                    ("old-seed", "old-result", "tag_similarity", 1.0, "{}", "old"),
                ],
            )
            db.executemany(
                "INSERT INTO node_pagerank VALUES (?, ?, ?, ?)",
                [
                    ("work-b", "similar-work", 0.20, "b1"),
                    ("work-c", "similar-work", 0.70, "b1"),
                    ("wrong-kind", "similar-work", 0.99, "b1"),
                    ("old-result", "similar-work", 1.0, "old"),
                ],
            )
        self.searcher = OfflineGraphSearch(self.database)

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_is_simple_and_does_not_read_graph_tables(self):
        results = self.searcher.search(
            "seed",
            vector_candidates=[("episode_card", "flat-result", 0.8)],
        )
        self.assertEqual([result.entity_id for result in results], ["flat-result"])
        self.assertIsNone(results[0].graph_score)
        self.assertIsNone(results[0].fusion_score)
        self.assertEqual(results[0].provenance[0]["source"], "vector")

    def test_named_task_filters_edges_without_explicit_edge_kinds(self):
        results = self.searcher.search("seed", mode="graph", task="similar_work", build_id="b1")
        self.assertEqual({row.entity_id for row in results}, {"work-b", "work-c"})

    def test_legacy_credit_projection_is_not_silently_used_as_heterogeneous(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("CREATE TABLE task_pagerank(seed_entity_id TEXT,node_entity_id TEXT,"
                       "graph_kind TEXT,rank INTEGER,score REAL,evidence_json TEXT,build_id TEXT)")
            db.execute("INSERT INTO task_pagerank VALUES('seed','work-b','voice_cast',1,0.5,"
                       "'{\"edge_kinds\":[\"cv_overlap\"],\"restart\":0.15}','b1')")
        with self.assertRaisesRegex(ValueError, "legacy overlap projection"):
            self.searcher.search("seed", mode="graph", task="voice_cast", build_id="b1")
        # Even though a staff_overlap derived edge exists, an unmaterialized
        # staff seed must not use the old one-hop subject projection.
        self.assertEqual(self.searcher.search("seed", mode="graph", task="staff_filmography", build_id="b1"), [])

    def test_materialized_restart_cannot_be_relabelled_at_query_time(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("CREATE TABLE task_pagerank(seed_entity_id TEXT,node_entity_id TEXT,"
                       "graph_kind TEXT,rank INTEGER,score REAL,evidence_json TEXT,build_id TEXT)")
            db.execute("INSERT INTO task_pagerank VALUES('seed','work-b','similar_work',1,0.1,"
                       "'{\"restart\":0.15,\"edge_kinds\":[\"tag_similarity\",\"same_season\"]}','b1')")
        results = self.searcher.search("seed", mode="graph", task="similar_work", build_id="b1")
        self.assertEqual([row.entity_id for row in results], ["work-b"])
        with self.assertRaisesRegex(ValueError, "restart"):
            self.searcher.search("seed", mode="graph", task="similar_work", build_id="b1", restart=0.2)

    def test_graph_reads_undirected_task_slice_and_build(self):
        results = self.searcher.search(
            "seed",
            mode="graph",
            task=GraphTask(
                name="similar-work",
                graph_kind="similar-work",
                edge_kinds=("tag_similarity",),
            ),
            build_id="b1",
        )
        self.assertEqual([result.entity_id for result in results], ["work-c", "work-b"])
        self.assertEqual(results[0].kind, "tag_similarity")
        self.assertEqual(results[0].raw_score, 0.8)
        self.assertEqual(results[0].graph_score, 0.7)
        self.assertEqual(results[0].provenance[0]["source"], "derived_edge")
        self.assertEqual(results[0].provenance[1]["source"], "node_pagerank")

    def test_fusion_accepts_injected_vector_and_lexical_candidates(self):
        results = self.searcher.search(
            "seed",
            mode="fusion",
            graph_kind="similar-work",
            build_id="b1",
            vector_candidates=[{"entity_id": "work-b", "kind": "episode_card", "score": 0.9}],
            lexical_candidates=[{"entity_id": "work-c", "kind": "fts", "score": 4.0}],
            top_k=DEFAULT_TOP_K,
        )
        by_id = {result.entity_id: result for result in results}
        self.assertIn("work-b", by_id)
        self.assertIn("work-c", by_id)
        self.assertEqual(by_id["work-b"].kind, "episode_card")
        self.assertEqual(by_id["work-c"].kind, "fts")
        self.assertEqual(by_id["work-c"].graph_score, 0.7)
        self.assertTrue(any(item["source"] == "fusion" for item in by_id["work-c"].provenance))
        self.assertTrue(all(result.fusion_score is not None for result in results))

    def test_graph_mode_requires_build_and_prefers_seed_personalized_rows(self):
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute(
                "CREATE TABLE task_pagerank(seed_entity_id TEXT,node_entity_id TEXT,"
                "graph_kind TEXT,rank INTEGER,score REAL,evidence_json TEXT,build_id TEXT)"
            )
            db.execute(
                "INSERT INTO task_pagerank VALUES('seed','work-d','similar-work',1,0.95,'{}','b1')"
            )
        with self.assertRaises(ValueError):
            self.searcher.search("seed", mode="graph", graph_kind="similar-work")
        results = self.searcher.search(
            "seed",
            mode="graph",
            graph_kind="similar-work",
            edge_kinds=("tag_similarity",),
            build_id="b1",
        )
        self.assertEqual(results[0].entity_id, "work-d")
        self.assertEqual(results[0].provenance[0]["source"], "task_pagerank")


class ABExperimentPackageTests(unittest.TestCase):
    def test_package_has_30_anonymous_unrated_queries(self):
        package = build_ab_package(
            [
                {
                    "query": f"query {index}",
                    "seed_entity_ids": [f"seed-{index}"],
                    "user_id": "omit-me",
                }
                for index in range(30)
            ]
        )
        payload = package.to_dict()
        self.assertEqual(len(package.queries), 30)
        self.assertEqual(payload["arms"], {"a": "simple", "b": "fusion"})
        self.assertNotIn("user_id", json.dumps(payload))
        self.assertNotIn("rating", payload)
        self.assertNotIn("ratings", payload)


if __name__ == "__main__":
    unittest.main()

import unittest

from mikirin.graph import (
    GraphEdge,
    aggregate_implicit_edge_weights,
    aggregate_staff_overlap,
    combine_scores,
    count_distinct_author_hashes,
    fuse_scores,
    pagerank_scores,
    rank_scores,
    subject_tag_similarities,
    tag_idf_weights,
    tag_similarity_edges,
    weighted_personalized_pagerank,
)
from mikirin.graph.algorithms import PageRankConvergenceError


class TagAndOverlapTests(unittest.TestCase):
    def test_idf_and_pairwise_similarity_are_order_independent(self) -> None:
        documents = {
            "s2": ["common", "rare"],
            "s1": ["common", "rare"],
            "s3": ["common", "other"],
        }
        reversed_documents = dict(reversed(list(documents.items())))

        self.assertEqual(tag_idf_weights(documents), tag_idf_weights(reversed_documents))
        weights = tag_idf_weights(documents)
        self.assertEqual(weights["common"], 0.0)
        self.assertGreater(weights["rare"], weights["common"])
        self.assertEqual(
            subject_tag_similarities(documents), subject_tag_similarities(reversed_documents)
        )
        self.assertGreater(subject_tag_similarities(documents)[("s1", "s2")], 0.9)

    def test_tag_edges_do_not_materialize_zero_weight_generic_tags(self) -> None:
        edges = tag_similarity_edges({"a": ["generic"], "b": ["generic"]})
        self.assertEqual(edges, ())

    def test_staff_overlap_deduplicates_members_and_is_sorted(self) -> None:
        edges = aggregate_staff_overlap(
            [
                ("b", "staff-2"),
                ("a", "staff-2"),
                ("a", "staff-2"),
                ("b", "staff-1"),
                ("a", "staff-1"),
            ]
        )
        self.assertEqual(edges, (GraphEdge("a", "b", 2.0, "staff_overlap"),))


class ImplicitAndPagerankTests(unittest.TestCase):
    def test_author_count_excludes_anonymous_and_empty_values(self) -> None:
        self.assertEqual(
            count_distinct_author_hashes(
                ["alice", "alice", "bob", "anonymous", "ANONYMOUS", "", "  ", None]
            ),
            2,
        )
        self.assertEqual(
            aggregate_implicit_edge_weights(
                [
                    {"src": "a", "dst": "b", "author_hash": "alice"},
                    {"src": "a", "dst": "b", "author_hash": "alice"},
                    {"src": "a", "dst": "b", "author_hash": "anonymous"},
                    {"src": "a", "dst": "b", "author_hash": "bob"},
                ]
            ),
            {("a", "b"): 2},
        )

    def test_weighted_pagerank_is_deterministic_and_supports_task_subgraph(self) -> None:
        edges = [
            GraphEdge("b", "c", 1.0),
            GraphEdge("a", "b", 3.0),
            GraphEdge("outside", "a", 100.0),
        ]
        reverse_edges = tuple(reversed(edges))
        first = weighted_personalized_pagerank(
            edges, {"a": 1.0}, task_nodes={"a", "b", "c"}, top_k=2
        )
        second = weighted_personalized_pagerank(
            reverse_edges, {"a": 1.0}, task_nodes={"c", "b", "a"}, top_k=2
        )
        self.assertEqual(first, second)
        self.assertEqual([node for node, _ in first], ["b", "a"])
        self.assertAlmostEqual(
            sum(
                pagerank_scores(
                    edges,
                    {"a": 1.0},
                    task_nodes={"a", "b", "c"},
                    max_iterations=1000,
                ).values()
            ),
            1.0,
        )

    def test_pagerank_ties_use_node_id(self) -> None:
        self.assertEqual(
            weighted_personalized_pagerank([], {"b": 1.0, "a": 1.0}, top_k=2),
            (("a", 0.5), ("b", 0.5)),
        )

    def test_pagerank_scores_reports_convergence_diagnostics(self) -> None:
        scores = pagerank_scores(
            [GraphEdge("a", "b", 1.0)],
            {"a": 1.0},
            max_iterations=1000,
            tolerance=1e-12,
        )

        self.assertEqual(set(scores), {"a", "b"})
        self.assertAlmostEqual(sum(scores.values()), 1.0)

    def test_pagerank_scores_raises_when_iteration_budget_is_exhausted(self) -> None:
        with self.assertRaises(PageRankConvergenceError) as raised:
            pagerank_scores(
                [GraphEdge("a", "b", 1.0)],
                {"a": 1.0},
                max_iterations=1,
                tolerance=0.0,
            )

        error = raised.exception
        self.assertEqual(error.iterations, 1)
        self.assertEqual(error.max_iterations, 1)
        self.assertEqual(error.tolerance, 0.0)
        self.assertGreater(error.last_delta, error.tolerance)
        self.assertIn("did not converge", str(error))


class FusionTests(unittest.TestCase):
    def test_three_score_strategies_and_stable_ranking(self) -> None:
        vectors = {"b": 0.4, "a": 0.8}
        graph = {"c": 0.7, "a": 0.2}
        self.assertEqual(
            combine_scores(vectors, graph, strategy="simple"), {"a": 0.8, "b": 0.4}
        )
        self.assertEqual(combine_scores(vectors, graph), {"a": 0.8, "b": 0.4})
        self.assertEqual(
            combine_scores(vectors, graph, strategy="graph"), {"a": 0.2, "c": 0.7}
        )
        self.assertEqual(
            combine_scores(vectors, graph, strategy="fusion", vector_weight=0.5, graph_weight=0.5),
            {"a": 0.5, "b": 0.2, "c": 0.35},
        )
        self.assertEqual(fuse_scores(vectors, graph), {"a": 0.5, "b": 0.2, "c": 0.35})
        self.assertEqual(rank_scores({"b": 1.0, "a": 1.0}), (("a", 1.0), ("b", 1.0)))


if __name__ == "__main__":
    unittest.main()

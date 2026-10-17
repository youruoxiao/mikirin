import unittest

from mikirin.slang.linking import (
    CONFIRMED,
    CROSS_WORK,
    PROVISIONAL,
    REJECT,
    SELF_REFERENCE,
    SINGLE_CHAR,
    decide_ingestion,
    link_candidates,
)


class SlangLinkingTests(unittest.TestCase):
    def test_self_reference_uses_concentration_and_official_name_exclusion(self):
        links = link_candidates(
            "团长",
            subject_anchors={
                "s2": {"anchor_count": 1, "document_count": 1, "author_count": 1},
                "s1": {"anchor_count": 9, "document_count": 9, "author_count": 25},
            },
            official_names={"s1": ["凉宫春日"], "s2": ["另一部作品"]},
        )

        self.assertEqual([item.entity_id for item in links], ["s1"])
        self.assertEqual(links[0].bucket, SELF_REFERENCE)
        self.assertEqual(links[0].rank, 1)
        self.assertEqual(links[0].evidence["anchor_concentration"], 0.9)
        self.assertEqual(links[0].evidence["official_name_exclusion"], 1.0)
        self.assertEqual(decide_ingestion(links[0]).decision, CONFIRMED)

    def test_official_name_is_not_an_alias_replacement(self):
        links = link_candidates(
            "凉宫春日",
            subject_anchors={"s1": {"anchor_count": 10, "document_count": 10}},
            official_names={"s1": ["凉宫春日"]},
        )
        self.assertEqual(len(links), 1)
        self.assertTrue(links[0].evidence["candidate_matches_official_name"])
        self.assertEqual(decide_ingestion(links[0]).decision, REJECT)

    def test_cross_work_consumes_external_top_three_only(self):
        calls = []

        class Retriever:
            def search(self, query, *, top_k):
                calls.append((query, top_k))
                return [
                    ("subject", "s3", 0.70),
                    ("subject", "s1", 0.90),
                    ("subject", "s4", 0.60),
                    ("subject", "s2", 0.80),
                ]

        links = link_candidates("团长", vector_retriever=Retriever())
        self.assertEqual(calls, [("团长", 3)])
        self.assertEqual([item.entity_id for item in links], ["s1", "s2", "s3"])
        self.assertTrue(all(item.bucket == CROSS_WORK for item in links))
        self.assertEqual([item.evidence["vector_rank"] for item in links], [1, 2, 3])

    def test_single_character_has_independent_conservative_threshold(self):
        links = link_candidates("锯", vector_neighbors=[{"entity_id": "s1", "score": 0.89}])
        self.assertEqual(links[0].bucket, SINGLE_CHAR)
        self.assertEqual(decide_ingestion(links[0]).decision, REJECT)

        medium = link_candidates("锯", vector_neighbors=[{"entity_id": "s1", "score": 0.94}])[0]
        self.assertEqual(decide_ingestion(medium).decision, PROVISIONAL)

        high = link_candidates("锯", vector_neighbors=[{"entity_id": "s1", "score": 0.96}])[0]
        self.assertEqual(decide_ingestion(high, author_breadth=21).decision, CONFIRMED)
        self.assertEqual(decide_ingestion(high, author_breadth=20).decision, REJECT)

    def test_tied_scores_are_stable_and_rejected_as_ambiguous(self):
        first = link_candidates(
            "团长",
            vector_neighbors=[
                {"entity_id": "s2", "score": 0.8},
                {"entity_id": "s1", "score": 0.8},
            ],
        )
        second = link_candidates(
            "团长",
            vector_neighbors=[
                {"entity_id": "s1", "score": 0.8},
                {"entity_id": "s2", "score": 0.8},
            ],
        )
        self.assertEqual([item.entity_id for item in first], ["s1", "s2"])
        self.assertEqual([item.entity_id for item in first], [item.entity_id for item in second])
        self.assertTrue(all(item.evidence["ambiguous"] for item in first))
        self.assertTrue(all(decide_ingestion(item).decision == REJECT for item in first))

    def test_empty_inputs_and_mapping_output(self):
        self.assertEqual(link_candidates(""), ())
        self.assertEqual(link_candidates(None, vector_neighbors=[]), ())
        self.assertEqual(decide_ingestion(None).effective_weight, 0.0)

        links = link_candidates(
            {"candidate_id": "c1", "candidate_text": "团长"},
            vector_neighbors=[{"entity_id": "s1", "score": 0.8}],
        )
        output = links[0].to_dict()
        self.assertEqual(output["candidate_id"], "c1")
        required = {"bucket", "rank", "score", "evidence"}
        self.assertEqual(required, set(output) & required)


if __name__ == "__main__":
    unittest.main()

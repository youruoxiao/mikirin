import json
import tempfile
import threading
import unittest
from pathlib import Path

from mikirin.graph.implicit import (
    EntityBrief,
    ImplicitCandidate,
    ImplicitRelationArbitrator,
    ImplicitRelationError,
    aggregate_implicit_edges,
    validate_implicit_response,
)


class FakeClient:
    model = "fake-implicit-model"
    provider = "fake"

    def __init__(self, responses=None, factory=None):
        self.responses = list(responses or [])
        self.factory = factory
        self.calls = 0
        self.messages = []
        self._lock = threading.Lock()

    def complete_json(self, messages, *, max_tokens=4096, reasoning_effort="high"):
        with self._lock:
            self.calls += 1
            self.messages.append(messages)
            payload = self.factory(messages) if self.factory is not None else self.responses.pop(0)
        return payload


def candidate(candidate_id="doc:1", *, author_hash="alice"):
    return ImplicitCandidate(
        candidate_id=candidate_id,
        document_id=candidate_id,
        text="Alice and Bob are close friends in the story.",
        entities=(
            EntityBrief("entity:b", "Bob", "A character named Bob."),
            EntityBrief("entity:a", "Alice", "A character named Alice."),
        ),
        author_hash=author_hash,
    )


def relation_response(*, src="entity:b", dst="entity:a", quote="close friends"):
    return {
        "src": src,
        "dst": dst,
        "relation": "friendship",
        "sentiment": "positive",
        "confidence": 0.8,
        "claim_quote": quote,
    }


class ImplicitPassBTests(unittest.TestCase):
    def test_closed_set_output_orders_pair_and_attaches_quote_provenance(self):
        item = candidate()
        result = ImplicitRelationArbitrator(client=FakeClient([relation_response()])).arbitrate(item)

        self.assertEqual(
            {key: result[key] for key in relation_response()},
            {
                "src": "entity:a",
                "dst": "entity:b",
                "relation": "friendship",
                "sentiment": "positive",
                "confidence": 0.8,
                "claim_quote": "close friends",
            },
        )
        self.assertEqual(result["provenance"]["document_id"], item.document_id)
        self.assertEqual(result["provenance"]["span"], {"start": 18, "end": 31})
        self.assertTrue(result["provenance"]["cache_key"])

    def test_reject_is_strict_and_invalid_fields_are_retried(self):
        item = candidate()
        rejected = ImplicitRelationArbitrator(client=FakeClient([{"reject": True}])).arbitrate(item)
        self.assertEqual(rejected["reject"], "reject")
        self.assertEqual(rejected["provenance"]["decision"], "reject")

        bad = dict(relation_response())
        bad["relation"] = "invented"
        fake = FakeClient([bad, relation_response()])
        result = ImplicitRelationArbitrator(client=fake, max_attempts=2).arbitrate(item)
        self.assertEqual(fake.calls, 2)
        self.assertEqual(result["relation"], "friendship")

    def test_quote_validation_rejects_url_garbage_empty_and_non_substrings(self):
        item = candidate()
        for quote in ("", "https://example.com/x", "�", "not in document"):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                validate_implicit_response(
                    {**relation_response(), "claim_quote": quote}, item
                )

    def test_cache_key_retry_and_cache_hit_are_offline(self):
        item = candidate()
        with tempfile.TemporaryDirectory() as directory:
            first_client = FakeClient(["```json\n{}\n```", relation_response()])
            first = ImplicitRelationArbitrator(
                client=first_client,
                cache=Path(directory),
                max_attempts=2,
            )
            first_result = first.arbitrate(item)
            second_client = FakeClient()
            second = ImplicitRelationArbitrator(client=second_client, cache=Path(directory))
            second_result = second.arbitrate(item)

            self.assertEqual(first.cache_key(item), second.cache_key(item))
            self.assertEqual(first_client.calls, 2)
            self.assertEqual(second_client.calls, 0)
            self.assertFalse(first_result["provenance"]["cache_hit"])
            self.assertTrue(second_result["provenance"]["cache_hit"])

    def test_concurrent_many_preserves_input_order(self):
        items = [candidate(f"doc:{index}") for index in range(3)]

        def response(messages):
            payload = json.loads(messages[1]["content"].split("\n", 1)[1])
            text = payload["candidate"]["text"]
            return relation_response(quote="close friends") if text else {"reject": True}

        fake = FakeClient(factory=response)
        results = ImplicitRelationArbitrator(client=fake, workers=3).arbitrate_many(items)
        self.assertEqual([result["provenance"]["candidate_id"] for result in results], [
            "doc:0", "doc:1", "doc:2"
        ])
        self.assertEqual(fake.calls, 3)


class ImplicitAggregationTests(unittest.TestCase):
    def test_weight_is_distinct_non_anonymous_authors_and_confidence_is_average(self):
        observations = [
            {**relation_response(), "author_hash": "alice", "source": "bgm"},
            {**relation_response(src="entity:a", dst="entity:b"), "author_hash": "alice"},
            {**relation_response(), "author_hash": "anonymous"},
            {**relation_response(), "author_hash": "", "confidence": 0.6},
            {**relation_response(), "author_hash": "bob", "confidence": 0.7},
            {"reject": "reject", "author_hash": "carol"},
        ]
        result = aggregate_implicit_edges(observations)

        self.assertEqual(len(result), 1)
        edge = result[0]
        self.assertEqual((edge["src"], edge["dst"]), ("entity:a", "entity:b"))
        self.assertEqual(edge["weight"], 2)
        self.assertEqual(edge["evidence_count"], 5)
        self.assertAlmostEqual(edge["confidence"], 0.74)
        self.assertIn("2 位不同作者", edge["attribution"])
        self.assertIn("5 条证据", edge["attribution"])

    def test_aggregation_rejects_invalid_evidence_instead_of_silently_writing_it(self):
        invalid = relation_response()
        invalid["sentiment"] = "maybe"
        with self.assertRaises(ValueError):
            aggregate_implicit_edges([invalid])


if __name__ == "__main__":
    unittest.main()

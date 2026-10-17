import json
import threading
import tempfile
import unittest
from pathlib import Path

from mikirin.cache import ModelCache
from mikirin.slang.arbitration import (
    ArbitrationError,
    CommentContext,
    EntityBrief,
    SlangArbitrator,
    SlangCandidate,
    parse_json_strict,
)


class FakeClient:
    """Offline structured client; it never has a network-capable code path."""

    model = "fake-slang-model"
    provider = "fake"

    def __init__(self, responses=None, factory=None) -> None:
        self.responses = list(responses or [])
        self.factory = factory
        self.calls = 0
        self.messages = []
        self._lock = threading.Lock()

    def complete_json(self, messages, *, max_tokens=4096, reasoning_effort="high"):
        with self._lock:
            self.calls += 1
            self.messages.append(messages)
            if self.factory is not None:
                payload = self.factory(messages)
            elif self.responses:
                payload = self.responses.pop(0)
            else:
                raise AssertionError("fake client was called more than expected")
        content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return {"choices": [{"message": {"content": content}}]}

    @staticmethod
    def response_content(payload):
        return payload["choices"][0]["message"]["content"]


def candidate(candidate_id="slang:1", *, slang="锯") -> SlangCandidate:
    return SlangCandidate(
        candidate_id=candidate_id,
        slang=slang,
        comments=(
            CommentContext(
                comment_id=f"comment:{candidate_id}",
                content="这部作品里大家说的锯明显是在指电锯人。",
            ),
        ),
        entities=(
            EntityBrief(
                entity_id="entity:chainsaw-man",
                name="电锯人",
                summary="一部以电锯恶魔与少年主角为核心的漫画及动画作品。",
            ),
            EntityBrief(
                entity_id="entity:other",
                name="其他作品",
                summary="另一个不应被默认选中的候选实体。",
            ),
        ),
    )


def decision_for(item: SlangCandidate, decision="entity:chainsaw-man", quote="锯"):
    return {
        "candidate_id": item.candidate_id,
        "decision": decision,
        "confidence": 0.93,
        "comment_id": item.comments[0].comment_id,
        "quote": quote,
        "reason": "评论明确把黑话用于该实体",
    }


class SlangArbitrationTests(unittest.TestCase):
    def test_closed_set_prompt_and_provenance_are_preserved(self) -> None:
        item = candidate()
        fake = FakeClient(responses=[{"decisions": [decision_for(item)]}])
        result = SlangArbitrator(client=fake).arbitrate(item)

        output = result["decisions"][0]
        self.assertEqual(output["decision"], "entity:chainsaw-man")
        self.assertEqual(output["quote"], "锯")
        self.assertEqual(output["provenance"]["source_kind"], "comment")
        self.assertEqual(output["provenance"]["comment_id"], item.comments[0].comment_id)
        self.assertEqual(output["provenance"]["quote"], "锯")
        self.assertEqual(output["provenance"]["span"], {"start": 9, "end": 10})
        self.assertIn('"entity_id":"entity:chainsaw-man"', fake.messages[0][1]["content"])
        self.assertIn('"entity_id":"entity:other"', fake.messages[0][1]["content"])
        self.assertIn('"reject"', fake.messages[0][0]["content"])

    def test_reject_is_a_valid_closed_set_decision(self) -> None:
        item = candidate()
        response = decision_for(item, decision="reject", quote="大家说的锯")
        result = SlangArbitrator(client=FakeClient(responses=[{"decisions": [response]}])).arbitrate(item)
        self.assertEqual(result["decisions"][0]["decision"], "reject")

    def test_invalid_json_and_unknown_entity_are_retried(self) -> None:
        item = candidate()
        valid = {"decisions": [decision_for(item)]}
        fake = FakeClient(responses=["```json\n{}\n```", valid])
        result = SlangArbitrator(client=fake, max_attempts=2).arbitrate(item)
        self.assertEqual(fake.calls, 2)
        self.assertEqual(result["decisions"][0]["decision"], "entity:chainsaw-man")

        invalid = decision_for(item, decision="entity:not-in-candidate")
        fake = FakeClient(responses=[{"decisions": [invalid]}, valid])
        result = SlangArbitrator(client=fake, max_attempts=2).arbitrate(item)
        self.assertEqual(fake.calls, 2)
        self.assertEqual(result["decisions"][0]["decision"], "entity:chainsaw-man")

    def test_bad_quotes_are_rejected_and_never_cached(self) -> None:
        item = candidate()
        bad_quotes = ["", "https://example.com/x", "锟斤拷", "不在评论中的词"]
        for bad_quote in bad_quotes:
            with self.subTest(bad_quote=bad_quote):
                fake = FakeClient(responses=[{"decisions": [decision_for(item, quote=bad_quote)]}])
                with self.assertRaises(ArbitrationError) as raised:
                    SlangArbitrator(client=fake, max_attempts=1).arbitrate(item)
                self.assertEqual(raised.exception.candidate_ids, (item.candidate_id,))
                self.assertEqual(fake.calls, 1)

    def test_strict_json_rejects_markdown_duplicate_keys_and_nan(self) -> None:
        for content in (
            "```json\n{}\n```",
            '{"decisions":[],"decisions":[]}',
            '{"decisions":[{"confidence":NaN}]}',
        ):
            with self.subTest(content=content), self.assertRaises((ValueError, json.JSONDecodeError)):
                parse_json_strict(content)

    def test_cache_key_is_deterministic_and_second_run_is_recoverable(self) -> None:
        item = candidate()
        with tempfile.TemporaryDirectory() as directory:
            cache = ModelCache(Path(directory))
            first_client = FakeClient(responses=[{"decisions": [decision_for(item)]}])
            first = SlangArbitrator(client=first_client, cache=cache)
            first_result = first.arbitrate(item)

            second_client = FakeClient()
            second = SlangArbitrator(client=second_client, cache=cache)
            second_result = second.arbitrate(item)

            self.assertEqual(first.cache_key([item]), second.cache_key([item]))
            self.assertEqual(first_client.calls, 1)
            self.assertEqual(second_client.calls, 0)
            self.assertFalse(first_result["decisions"][0]["provenance"]["cache_hit"])
            self.assertTrue(second_result["decisions"][0]["provenance"]["cache_hit"])
            self.assertEqual(
                first_result["decisions"][0]["quote"],
                second_result["decisions"][0]["quote"],
            )

    def test_batch_interface_preserves_order_and_uses_workers(self) -> None:
        items = [candidate(f"slang:{index}", slang=f"锯{index}") for index in range(3)]

        def make_response(messages):
            payload = json.loads(messages[1]["content"].split("\n", 1)[1])
            return {
                "decisions": [
                    decision_for(
                        SlangCandidate.from_dict(raw),
                        quote="锯",
                    )
                    for raw in payload["candidates"]
                ]
            }

        fake = FakeClient(factory=make_response)
        result = SlangArbitrator(client=fake, batch_size=2, workers=2).arbitrate_batches(items)
        self.assertEqual(
            [item["candidate_id"] for item in result["decisions"]],
            [item.candidate_id for item in items],
        )
        self.assertEqual(fake.calls, 2)

    def test_input_schema_rejects_extra_fields_and_empty_context(self) -> None:
        with self.assertRaises(ValueError):
            SlangCandidate.from_dict(
                {
                    "candidate_id": "s1",
                    "slang": "锯",
                    "comments": [],
                    "entities": [],
                    "extra": "not allowed",
                }
            )
        with self.assertRaises(ValueError):
            SlangCandidate("s1", "锯", (), (EntityBrief("e1", "作品", "简介"),))


if __name__ == "__main__":
    unittest.main()

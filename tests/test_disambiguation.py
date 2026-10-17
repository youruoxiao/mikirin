import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from mikirin.cache import ModelCache
from mikirin.disambiguation import ClosedCandidateDisambiguator, ClosedStanceClassifier


class DisambiguationTests(unittest.TestCase):
    def test_stance_is_closed_quoted_and_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Mock()
            label = {'candidate_id': 'c1', 'stance': 'dissenting', 'claim_quote': '节奏太赶'}
            client.response_content.return_value = json.dumps({'labels': [label]})
            classifier = ClosedStanceClassifier(client, ModelCache(Path(directory)),
                model='test', provider='test', base_url='https://example.test/v1')
            candidates = [{'candidate_id': 'c1', 'content': '我觉得节奏太赶了'}]
            self.assertEqual(classifier('如何评价', candidates), [label])
            self.assertEqual(classifier('如何评价', candidates), [label])
            self.assertEqual(client.complete_json.call_count, 1)
            label['claim_quote'] = '原文没有这句话'
            client.response_content.return_value = json.dumps({'labels': [label]})
            with self.assertRaises(ValueError):
                classifier('另一问题', candidates)

    def test_cache_closed_set_abstention_and_provider_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Mock()
            client.response_content.return_value = json.dumps({'episode_entity_id': 'e2'})
            cache = ModelCache(Path(directory))
            selector = ClosedCandidateDisambiguator(client, cache, model='test', provider='test', base_url='https://example.test/v1')
            candidates = [{'episode_entity_id': 'e1'}, {'episode_entity_id': 'e2'}]
            self.assertEqual(selector('哪集', candidates), 'e2')
            self.assertEqual(selector('哪集', candidates), 'e2')
            self.assertEqual(client.complete_json.call_count, 1)
            client.response_content.return_value = '{"episode_entity_id": null}'
            other = ClosedCandidateDisambiguator(client, cache, model='test', provider='test', base_url='https://other.test/v1')
            self.assertIsNone(other('哪集', candidates))
            self.assertEqual(client.complete_json.call_count, 2)
            client.response_content.return_value = '{"episode_entity_id":"hallucination"}'
            with self.assertRaisesRegex(ValueError, 'outside'):
                selector('新问题', candidates)


if __name__ == '__main__':
    unittest.main()

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mikirin.cli import build_parser
from mikirin.demo import create_demo
from mikirin.layout import ProjectLayout


class FeatureCLITests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        create_demo(ProjectLayout(self.root))

    def command(self, *arguments):
        args = build_parser().parse_args(['--root', str(self.root), *arguments])
        out = io.StringIO()
        with redirect_stdout(out), patch('urllib.request.urlopen', side_effect=AssertionError('offline')):
            self.assertEqual(args.handler(args), 0)
        return json.loads(out.getvalue())

    def test_simple_graph_facade_is_offline_and_retrieves_demo(self):
        result = self.command('graph-search', '放映部')
        self.assertFalse(result['experimental'])
        self.assertEqual(result['results'][0]['entity_id'], 'demo:subject:rooftop')

    def test_graph_requires_explicit_build(self):
        args = build_parser().parse_args(['--root', str(self.root), 'graph-search', '放映部', '--mode', 'fusion'])
        with self.assertRaisesRegex(ValueError, 'build-id'):
            args.handler(args)

    def test_alias_review_never_overwrites_user_labels(self):
        destination = self.root / 'review.json'
        self.command('p1-review', 'aliases-prepare', '--month', '2026-10', '--output', str(destination))
        destination.write_text('[{"label": true}]', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            self.command('p1-review', 'aliases-prepare', '--month', '2026-10', '--output', str(destination))
        self.assertEqual(json.loads(destination.read_text()), [{'label': True}])

    def test_rss_is_disabled_and_subscription_does_not_fetch(self):
        result = self.command('rss', 'status')
        self.assertFalse(result['settings']['enabled'])
        self.assertEqual(result['settings']['probability'], .3)
        subscribed = self.command('rss', 'subscribe', '放映部', 'https://example.test/feed.xml')
        self.assertEqual(subscribed['subject_id'], 'demo:subject:rooftop')
        self.assertTrue(self.command('rss', 'poll')['disabled'])
        self.assertTrue(self.command('rss', 'unsubscribe', subscribed['id'])['unsubscribed'])

    def test_empty_memory_queue_needs_no_api_credentials(self):
        self.assertEqual(self.command('memory-queue', 'status'), [])
        self.assertEqual(self.command('memory-queue', 'run')['reports'], [])

    def test_research_search_is_explicit_and_all_modes_work_offline(self):
        for mode in ('fts', 'anchored', 'evidence', 'hybrid'):
            with self.subTest(mode=mode):
                result = self.command('research-search', '放映部', '--mode', mode)
                self.assertTrue(result['experimental'])
                if mode != 'fts':
                    self.assertIn('demo:subject:rooftop', [row['entity_id'] for row in result['entities']])

    def test_retrieval_benchmark_report_has_no_fabricated_gold(self):
        benchmark = self.root / 'queries.json'
        benchmark.write_text(json.dumps({'version':'test-v1', 'questions':[
            {'id':'one','query':'放映部','category':'factual','gold':None}
        ]}), encoding='utf-8')
        report = self.root / 'evaluation.json'
        self.command('retrieval-eval', '--benchmark', str(benchmark), '--output', str(report))
        data = json.loads(report.read_text(encoding='utf-8'))
        self.assertEqual(len(data['results']), 4)
        self.assertFalse(data['human_approved'])
        self.assertTrue(all(row['metrics']['scope_accuracy'] is None for row in data['results']))


if __name__ == '__main__':
    unittest.main()

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from mikirin.demo import create_demo
from mikirin.layout import ProjectLayout
from mikirin.query import KnowledgeQueries
from mikirin.search_index import SearchIndexer


class MinedAliasTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.database = create_demo(ProjectLayout(root))
        with closing(sqlite3.connect(self.database)) as db, db:
            for i, (slug, status, confidence, text) in enumerate((
                ('rooftop', 'confirmed', .7, '天台电影'),
                ('radio', 'provisional', 1., '天台电影'),
                ('last-train', 'rejected', 1., '错误昵称'),
                ('last-train', 'candidate', 1., '未经审核'),
            )):
                db.execute(
                    "INSERT INTO entity_alias(id,entity_id,alias_text,normalized_alias,"
                    "source_kind,status,confidence,source_asset_id,build_id,license_id) "
                    "VALUES(?,?,?,?,'mined',?,?,'demo','demo-v1','demo-original')",
                    (f'mined:{i}', f'demo:subject:{slug}', text, text, status, confidence),
                )
        SearchIndexer.rebuild(self.database)

    def test_provisional_discount_applies_to_exact_and_contained_alias(self):
        for query in ('天台电影', '我想看看天台电影'):
            hits = KnowledgeQueries.search_entities(self.database, query)
            self.assertEqual(hits[0].entity_id, 'demo:subject:rooftop')
            self.assertIn('demo:subject:radio', [h.entity_id for h in hits])
            self.assertEqual(next(h for h in hits if h.entity_id == 'demo:subject:radio').retrieval_weight, .5)

    def test_rejected_and_unreviewed_mined_aliases_never_enter_fts(self):
        for query in ('错误昵称', '未经审核'):
            self.assertEqual(KnowledgeQueries.search_entities(self.database, query), [])
            with closing(sqlite3.connect(self.database)) as db:
                self.assertEqual(db.execute(
                    'SELECT count(*) FROM entity_fts WHERE entity_fts MATCH ?', (query,)
                ).fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()

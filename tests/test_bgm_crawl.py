import json
import sqlite3
import tempfile
import unittest
import urllib.error
from contextlib import closing
from pathlib import Path

from mikirin.layout import ProjectLayout
from mikirin.sources.bgm_crawl import BgmContentCrawler, BgmLinksParser, sampled_pages


class BgmCrawlTests(unittest.TestCase):
    def test_quantile_sampling_keeps_edges_and_cap(self) -> None:
        self.assertEqual(sampled_pages(4, 8), [1, 2, 3, 4])
        pages = sampled_pages(100, 8)
        self.assertEqual((pages[0], pages[-1], len(pages)), (1, 100, 8))

    def test_links_parser_finds_pages_and_blogs(self) -> None:
        parser = BgmLinksParser()
        parser.feed(
            '<a href="?page=12">12</a><a href="/subject/328609/reviews/29.html">29</a>'
            '<a href="/blog/456">review</a>'
        )
        self.assertEqual(parser.page_numbers, {12, 29})
        self.assertEqual(parser.blog_ids, {456})

    def test_seed_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = ProjectLayout(Path(directory))
            selection = Path(directory) / "selection.json"
            selection.write_text(json.dumps({
                "selection_id": "sample-1",
                "subjects": [{"subject_id": 123, "title": "A", "sources": []}],
            }), encoding="utf-8")
            crawler = BgmContentCrawler(layout)
            first = crawler.seed(selection)
            second = crawler.seed(selection)
            self.assertEqual(first["total"], 3)
            self.assertEqual(second["total"], 3)
            self.assertEqual(first["by_kind"]["episode_meta"]["pending"], 1)
            self.assertEqual(first["unfinished"], 3)

    def test_permanent_missing_page_is_not_retried(self) -> None:
        class MissingCrawler(BgmContentCrawler):
            def _request(self, url: str) -> tuple[bytes, int, str, str]:
                raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

        with tempfile.TemporaryDirectory() as directory:
            layout = ProjectLayout(Path(directory))
            selection = Path(directory) / "selection.json"
            selection.write_text(json.dumps({
                "selection_id": "sample-404",
                "subjects": [{"subject_id": 404, "title": "Missing", "sources": []}],
            }), encoding="utf-8")
            crawler = MissingCrawler(layout)
            crawler.seed(selection)
            first = crawler.crawl(limit=1)
            crawler.crawl(limit=1)
            with closing(sqlite3.connect(crawler.queue_db)) as db:
                attempts = db.execute(
                    "SELECT attempts FROM crawl_item WHERE key='episode-meta:404:0'"
                ).fetchone()[0]
            self.assertEqual(first["status_not_found"], 1)
            self.assertEqual(attempts, 1)


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from mikirin.layout import ProjectLayout
from mikirin.sources.bgm_crawl import BgmContentCrawler
from mikirin.sources.bgm_web_staging import (
    AuthorHasher,
    BgmWebStager,
    BlogArticleParser,
    CommunityCommentParser,
)


SUBJECT_COMMENTS_HTML = """
<div id="comment_box">
  <div class="item clearit" data-item-user="alice">
    <a href="/user/alice">Alice</a>
    <span class="starlight stars9"></span>
    <small class="grey">看过</small><small class="grey">@ 2026-7-27 12:00</small>
    <p class="comment">第一行<br>第二行</p>
    <div class="likes_grid" id="likes_grid_12345"></div>
  </div>
  <div class="item clearit">
    <a href="/user/bob">Bob</a>
    <p class="comment">另一条评论</p>
    <div class="likes_grid" id="likes_grid_67890"></div>
  </div>
</div>
"""


class BgmWebStagingTests(unittest.TestCase):
    def test_comment_parser_extracts_identity_rating_and_text(self) -> None:
        parser = CommunityCommentParser()
        parser.feed(SUBJECT_COMMENTS_HTML)
        self.assertEqual(len(parser.comments), 2)
        self.assertEqual(parser.comments[0].source_record_id, "12345")
        self.assertEqual(parser.comments[0].author_source_id, "alice")
        self.assertEqual(parser.comments[0].rating, 9)
        self.assertEqual(parser.comments[0].content, "第一行\n第二行")
        self.assertEqual(parser.comments[0].metadata_text, ("看过", "@ 2026-7-27 12:00"))

    def test_blog_parser_keeps_article_body(self) -> None:
        parser = BlogArticleParser()
        parser.feed(
            '<div class="author user-card"><a href="/user/writer">作者</a></div>'
            '<h1 class="title">长评标题</h1>'
            '<div class="time">2026-7-27 12:00 · 3 分钟阅读</div>'
            '<div id="entry_content"><p>第一段</p><p>第二段<br>续行</p></div>'
        )
        self.assertEqual(parser.title, "长评标题")
        self.assertEqual(parser.content, "第一段\n第二段\n续行")
        self.assertEqual(parser.author_source_id, "writer")
        self.assertEqual(parser.time_text, "2026-7-27 12:00 · 3 分钟阅读")

    def test_episode_parser_keeps_nested_replies_as_separate_documents(self) -> None:
        parser = CommunityCommentParser()
        parser.feed(
            '<div id="post_1" class="row row_reply clearit" data-item-user="parent">'
            '<div class="action"><small>#1 - 2026-7-27 12:00</small></div>'
            '<div class="reply_content"><div class="message clearit">主楼</div>'
            '<div id="post_2" class="sub_reply_bg clearit" data-item-user="child">'
            '<small>#1-1 - 2026-7-27 12:01</small>'
            '<div class="cmt_sub_content">楼中楼</div></div></div></div>'
        )
        self.assertEqual(len(parser.comments), 2)
        by_id = {comment.source_record_id: comment for comment in parser.comments}
        self.assertEqual(by_id["1"].content, "主楼")
        self.assertEqual(by_id["2"].content, "楼中楼")
        self.assertEqual(by_id["2"].author_source_id, "child")

    def test_author_hash_is_stable_and_source_id_is_not_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            salt = Path(directory) / "salt"
            first = AuthorHasher(salt).hash("alice")
            second = AuthorHasher(salt).hash("alice")
            self.assertEqual(first, second)
            self.assertNotIn("alice", first)
            self.assertEqual(len(first), 64)

    def test_stager_creates_immutable_partial_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = ProjectLayout(Path(directory))
            layout.initialize()
            crawler = BgmContentCrawler(layout)
            raw = SUBJECT_COMMENTS_HTML.encode()
            digest = hashlib.sha256(raw).hexdigest()
            raw_path = layout.raw / "bgm_web" / "2026-07-27" / f"{digest}.html"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(raw)
            now = "2026-07-27T00:00:00+00:00"
            with closing(crawler.connect()) as db, db:
                db.execute(
                    """
                    INSERT INTO crawl_item(
                        key, kind, subject_id, url, priority, status, attempts,
                        content_hash, relative_path, http_status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "subject-comments:42:1", "subject_comments", 42,
                        "https://bangumi.tv/subject/42/comments?page=1", 20, "ok", 1,
                        f"sha256:{digest}", raw_path.relative_to(layout.root).as_posix(),
                        200, now, now,
                    ),
                )
                db.execute(
                    """
                    INSERT INTO crawl_item(
                        key, kind, subject_id, url, priority, status, attempts,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "episode-comments:99", "episode_comments", 42,
                        "https://bangumi.tv/ep/99", 70, "pending", 0, now, now,
                    ),
                )
                db.commit()

            report = BgmWebStager(layout).stage(source_version="fixture")
            self.assertEqual(report.documents, 2)
            self.assertEqual(report.source_pages, 1)
            self.assertEqual(report.pending_pages, 1)
            self.assertTrue(report.partial)
            self.assertTrue(report.documents_hash.startswith("sha256:"))
            self.assertGreater(report.documents_bytes, 0)
            self.assertEqual(report.document_kinds, {"bgm_subject_comment": 2})
            self.assertEqual(report.missing_author_hashes, 0)
            lines = (layout.staging / "bgm_web" / "fixture" / "documents.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            documents = [json.loads(line) for line in lines]
            self.assertEqual(documents[0]["subject_entity_id"], "bgm:subject:42")
            self.assertEqual(documents[0]["source_kind"], "bgm_subject_comment")
            self.assertNotEqual(documents[0]["author_hash"], "alice")
            self.assertFalse(documents[0]["redistributable"])
            with self.assertRaises(FileExistsError):
                BgmWebStager(layout).stage(source_version="fixture")

    def test_require_complete_rejects_pending_content_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            layout = ProjectLayout(Path(directory))
            crawler = BgmContentCrawler(layout)
            now = "2026-07-27T00:00:00+00:00"
            with closing(crawler.connect()) as db, db:
                db.execute(
                    """
                    INSERT INTO crawl_item(
                        key, kind, subject_id, url, priority, status, attempts,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("blog-review:1", "blog_review", 42, "https://bangumi.tv/blog/1",
                     60, "pending", 0, now, now),
                )
                db.commit()
            with self.assertRaisesRegex(ValueError, "unfinished"):
                BgmWebStager(layout).stage(source_version="fixture", require_complete=True)


if __name__ == "__main__":
    unittest.main()

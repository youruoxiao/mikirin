from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import uuid
from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from mikirin.atomic import atomic_write_json, atomic_write_text
from mikirin.layout import ProjectLayout


@dataclass(frozen=True, slots=True)
class ParsedComment:
    source_record_id: str | None
    author_source_id: str | None
    content: str
    rating: int | None
    metadata_text: tuple[str, ...]


def _clean_text(pieces: list[str]) -> str:
    lines = []
    for line in "".join(pieces).replace("\u3000", " ").splitlines():
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


class CommunityCommentParser(HTMLParser):
    ITEM_CLASSES = (
        {"item", "clearit"},
        {"row_reply", "clearit"},
        {"sub_reply_bg", "clearit"},
        {"reply_item"},
    )
    CONTENT_CLASSES = {"comment", "message", "reply_content", "cmt_sub_content"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.comments: list[ParsedComment] = []
        self._items: list[_CommentState] = []

    def _start_item(self, attributes: dict[str, str | None]) -> None:
        element_id = attributes.get("id") or ""
        post = re.fullmatch(r"post_(\d+)", element_id)
        self._items.append(
            _CommentState(
                depth=1,
                author=attributes.get("data-item-user"),
                record_id=post.group(1) if post else None,
            )
        )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "div":
            for item in self._items:
                item.depth += 1
            if any(expected <= classes for expected in self.ITEM_CLASSES):
                self._start_item(attributes)
        if not self._items:
            return
        item = self._items[-1]
        href = attributes.get("href") or ""
        author = re.fullmatch(r"/user/([^/?#]+)", href)
        if author and item.author is None:
            item.author = author.group(1)
        for class_name in classes:
            rating = re.fullmatch(r"stars(10|[1-9])", class_name)
            if rating:
                item.rating = int(rating.group(1))
        element_id = attributes.get("id") or ""
        record = re.fullmatch(r"likes_grid_(\d+)", element_id)
        if record:
            item.record_id = record.group(1)
        if (tag == "p" and "comment" in classes) or classes & self.CONTENT_CLASSES:
            item.content_depth = 1
        elif item.content_depth and tag in {"div", "p"}:
            item.content_depth += 1
        if tag == "small":
            item.small_depth = 1
            item.small = []
        if tag == "br" and item.content_depth:
            item.content.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if not self._items:
            return
        item = self._items[-1]
        if item.content_depth and tag in {"div", "p"}:
            item.content_depth -= 1
        if item.small_depth and tag == "small":
            text = _clean_text(item.small)
            if text:
                item.metadata.append(text)
            item.small_depth = 0
            item.small = []
        if tag == "div":
            for open_item in self._items:
                open_item.depth -= 1
            while self._items and self._items[-1].depth == 0:
                completed = self._items.pop()
                content = _clean_text(completed.content)
                if not content:
                    continue
                self.comments.append(
                    ParsedComment(
                        completed.record_id,
                        completed.author,
                        content,
                        completed.rating,
                        tuple(completed.metadata),
                    )
                )

    def handle_data(self, data: str) -> None:
        if not self._items:
            return
        item = self._items[-1]
        if item.content_depth:
            item.content.append(data)
        if item.small_depth:
            item.small.append(data)


@dataclass(slots=True)
class _CommentState:
    depth: int
    author: str | None = None
    record_id: str | None = None
    rating: int | None = None
    content_depth: int = 0
    small_depth: int = 0
    content: list[str] = field(default_factory=list)
    small: list[str] = field(default_factory=list)
    metadata: list[str] = field(default_factory=list)


class BlogArticleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._inside = False
        self._depth = 0
        self._title_depth = 0
        self._author_card_depth = 0
        self._time_depth = 0
        self._pieces: list[str] = []
        self._title: list[str] = []
        self._time: list[str] = []
        self.author_source_id: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if self._author_card_depth and tag == "div":
            self._author_card_depth += 1
        elif tag == "div" and {"author", "user-card"} <= classes:
            self._author_card_depth = 1
        if self._author_card_depth and tag == "a" and self.author_source_id is None:
            author = re.fullmatch(r"/user/([^/?#]+)", attributes.get("href") or "")
            if author:
                self.author_source_id = author.group(1)
        if self._time_depth and tag == "div":
            self._time_depth += 1
        elif tag == "div" and "time" in classes:
            self._time_depth = 1
            self._time = []
        if not self._inside and tag == "div" and attributes.get("id") in {
            "entry_content", "blog_entry"
        }:
            self._inside = True
            self._depth = 1
            return
        if self._inside:
            if tag == "div":
                self._depth += 1
            if tag in {"p", "div", "h1", "h2", "li", "br"}:
                self._pieces.append("\n")
        if tag in {"h1", "h2"} and ("title" in classes or attributes.get("id") == "headerSubject"):
            self._title_depth = 1

    def handle_endtag(self, tag: str) -> None:
        if self._inside:
            if tag == "div":
                self._depth -= 1
                if self._depth == 0:
                    self._inside = False
            if tag in {"p", "div", "h1", "h2", "li"}:
                self._pieces.append("\n")
        if self._title_depth and tag in {"h1", "h2"}:
            self._title_depth = 0
        if tag == "div" and self._author_card_depth:
            self._author_card_depth -= 1
        if tag == "div" and self._time_depth:
            self._time_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._inside:
            self._pieces.append(data)
        if self._title_depth:
            self._title.append(data)
        if self._time_depth:
            self._time.append(data)

    @property
    def content(self) -> str:
        return _clean_text(self._pieces)

    @property
    def title(self) -> str:
        return _clean_text(self._title)

    @property
    def time_text(self) -> str:
        return _clean_text(self._time)


class AuthorHasher:
    def __init__(self, salt_path: Path) -> None:
        salt_path.parent.mkdir(parents=True, exist_ok=True)
        if not salt_path.exists():
            atomic_write_text(salt_path, secrets.token_hex(32))
        self.salt = salt_path.read_text(encoding="utf-8").strip()
        if len(self.salt) < 32:
            raise ValueError("author salt is missing or too short")

    def hash(self, source_user_id: str | None) -> str:
        if not source_user_id:
            return ""
        return hashlib.sha256(f"bangumi\0{self.salt}\0{source_user_id}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class BgmWebStagingReport:
    source_version: str
    documents: int
    documents_hash: str
    documents_bytes: int
    document_kinds: dict[str, int]
    missing_author_hashes: int
    source_pages: int
    empty_pages: int
    duplicates: int
    pending_pages: int
    partial: bool
    output_directory: str


class BgmWebStager:
    SOURCE_KINDS = {"subject_comments", "episode_comments", "blog_review"}

    def __init__(self, layout: ProjectLayout) -> None:
        self.layout = layout

    def stage(self, *, source_version: str, require_complete: bool = False) -> BgmWebStagingReport:
        queue_path = self.layout.manifests / "bgm_web" / "crawl-queue.sqlite"
        with closing(sqlite3.connect(queue_path)) as db:
            db.row_factory = sqlite3.Row
            placeholders = ",".join("?" for _ in self.SOURCE_KINDS)
            rows = db.execute(
                f"SELECT * FROM crawl_item WHERE kind IN ({placeholders}) AND status='ok' "
                "ORDER BY kind, key",
                tuple(sorted(self.SOURCE_KINDS)),
            ).fetchall()
            pending = int(db.execute(
                f"SELECT count(*) FROM crawl_item WHERE kind IN ({placeholders}) "
                "AND status NOT IN ('ok','not_found')",
                tuple(sorted(self.SOURCE_KINDS)),
            ).fetchone()[0])
        if require_complete and pending:
            raise ValueError(f"Bangumi web crawl still has {pending} unfinished content pages")
        target = self.layout.staging / "bgm_web" / source_version
        if target.exists():
            raise FileExistsError(f"bgm web staging already exists: {target}")
        temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir(parents=True, exist_ok=False)
        hasher = AuthorHasher(self.layout.user_data / "author_salt")
        seen: set[str] = set()
        document_kinds: Counter[str] = Counter()
        documents = empty_pages = duplicates = 0
        missing_author_hashes = 0
        try:
            with (temporary / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as output:
                for row in rows:
                    raw_path = self.layout.root / row["relative_path"]
                    raw = raw_path.read_bytes()
                    actual_hash = "sha256:" + hashlib.sha256(raw).hexdigest()
                    if actual_hash != row["content_hash"]:
                        raise ValueError(
                            f"raw Bangumi page hash mismatch: {row['key']} "
                            f"expected {row['content_hash']} got {actual_hash}"
                        )
                    html = raw.decode("utf-8", errors="replace")
                    records = self._records(row, html, hasher)
                    if not records:
                        empty_pages += 1
                    for record in records:
                        if record["document_id"] in seen:
                            duplicates += 1
                            continue
                        seen.add(record["document_id"])
                        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        documents += 1
                        document_kinds[str(record["source_kind"])] += 1
                        missing_author_hashes += not bool(record.get("author_hash"))
            documents_path = temporary / "documents.jsonl"
            digest = hashlib.sha256()
            with documents_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            documents_hash = f"sha256:{digest.hexdigest()}"
            documents_bytes = documents_path.stat().st_size
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        report = BgmWebStagingReport(
            source_version=source_version,
            documents=documents,
            documents_hash=documents_hash,
            documents_bytes=documents_bytes,
            document_kinds=dict(sorted(document_kinds.items())),
            missing_author_hashes=missing_author_hashes,
            source_pages=len(rows),
            empty_pages=empty_pages,
            duplicates=duplicates,
            pending_pages=pending,
            partial=bool(pending),
            output_directory=target.relative_to(self.layout.root).as_posix(),
        )
        atomic_write_json(target / "manifest.json", asdict(report))
        return report

    def _records(
        self, row: sqlite3.Row, html: str, hasher: AuthorHasher
    ) -> list[dict[str, Any]]:
        kind = str(row["kind"])
        subject_id = int(row["subject_id"])
        base = {
            "source": "bangumi_web",
            "subject_entity_id": f"bgm:subject:{subject_id}",
            "episode_entity_id": (
                f"bgm:episode:{int(row['object_id'])}"
                if kind == "episode_comments" and row["object_id"] is not None else None
            ),
            "url": row["url"],
            "fetched_at": row["updated_at"],
            "source_asset_hash": row["content_hash"],
            "raw_relative_path": row["relative_path"],
            "license_id": "bangumi-user-content",
            "redistributable": False,
            "attribution_required": True,
            "commercial_use_allowed": False,
        }
        if kind == "blog_review":
            parser = BlogArticleParser()
            parser.feed(html)
            content = parser.content
            if not content:
                return []
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            blog_id = int(row["object_id"])
            return [{
                **base,
                "document_id": f"bgm:blog:{blog_id}:{content_hash}",
                "source_kind": "bgm_review",
                "source_record_id": str(blog_id),
                "author_hash": hasher.hash(parser.author_source_id),
                "rating": None,
                "posted_at_raw": parser.time_text or None,
                "title": parser.title or f"Bangumi blog {blog_id}",
                "raw_content": content,
                "content_hash": f"sha256:{content_hash}",
            }]
        parser = CommunityCommentParser()
        parser.feed(html)
        source_kind = "bgm_ep_comment" if kind == "episode_comments" else "bgm_subject_comment"
        records = []
        for comment in parser.comments:
            author_hash = hasher.hash(comment.author_source_id)
            identity = "\0".join((
                source_kind,
                str(subject_id),
                str(row["object_id"] or ""),
                str(comment.source_record_id or ""),
                author_hash,
                comment.content,
            ))
            digest = hashlib.sha256(identity.encode()).hexdigest()
            records.append({
                **base,
                "document_id": f"bgm:web-comment:{digest}",
                "source_kind": source_kind,
                "source_record_id": comment.source_record_id,
                "author_hash": author_hash,
                "rating": comment.rating,
                "posted_at_raw": comment.metadata_text[-1] if comment.metadata_text else None,
                "metadata_text": list(comment.metadata_text),
                "title": "",
                "raw_content": comment.content,
                "content_hash": "sha256:" + hashlib.sha256(comment.content.encode()).hexdigest(),
            })
        return records

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import sqlite3
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from mikirin.layout import ProjectLayout
from mikirin.sources.bgm_selection import USER_AGENT


WEB_ROOT = "https://bangumi.tv"
API_ROOT = "https://api.bgm.tv"


def sampled_pages(total_pages: int, cap: int = 8) -> list[int]:
    if total_pages < 1 or cap < 1:
        return []
    if total_pages <= cap:
        return list(range(1, total_pages + 1))
    return sorted({1 + round(index * (total_pages - 1) / (cap - 1)) for index in range(cap)})


class BgmLinksParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.page_numbers: set[int] = set()
        self.blog_ids: set[int] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href") or ""
        parsed = urllib.parse.urlparse(href)
        query = urllib.parse.parse_qs(parsed.query)
        for value in query.get("page", []):
            if value.isdigit():
                self.page_numbers.add(int(value))
        match = re.fullmatch(r"/blog/(\d+)", parsed.path)
        if match:
            self.blog_ids.add(int(match.group(1)))
        review_page = re.fullmatch(r"/subject/\d+/reviews/(\d+)\.html", parsed.path)
        if review_page:
            self.page_numbers.add(int(review_page.group(1)))


class BgmContentCrawler:
    def __init__(
        self,
        layout: ProjectLayout,
        *,
        delay_seconds: float = 1.0,
        timeout_seconds: float = 30.0,
        max_attempts: int = 5,
        short_comment_page_cap: int = 8,
    ) -> None:
        if delay_seconds < 1.0:
            raise ValueError("Bangumi requests must be at least one second apart")
        self.layout = layout
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.short_comment_page_cap = short_comment_page_cap
        self._last_request_at = 0.0

    @property
    def queue_db(self) -> Path:
        return self.layout.manifests / "bgm_web" / "crawl-queue.sqlite"

    def connect(self) -> sqlite3.Connection:
        self.queue_db.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.queue_db)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""
            CREATE TABLE IF NOT EXISTS crawl_item(
                key TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                subject_id INTEGER NOT NULL,
                object_id INTEGER,
                url TEXT NOT NULL,
                priority INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                content_hash TEXT,
                relative_path TEXT,
                http_status INTEGER,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_bgm_crawl_next ON crawl_item(status, priority, key)")
        return db

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def _enqueue(
        self,
        db: sqlite3.Connection,
        *,
        key: str,
        kind: str,
        subject_id: int,
        url: str,
        priority: int,
        object_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = self._now()
        db.execute(
            """
            INSERT OR IGNORE INTO crawl_item(
                key, kind, subject_id, object_id, url, priority,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (key, kind, subject_id, object_id, url, priority,
             json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True), now, now),
        )

    def seed(self, selection_path: Path) -> dict[str, int]:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection_id = selection["selection_id"]
        with closing(self.connect()) as db, db:
            for item in selection["subjects"]:
                subject_id = int(item["subject_id"])
                self._enqueue(
                    db, key=f"episode-meta:{subject_id}:0", kind="episode_meta",
                    subject_id=subject_id,
                    url=f"{API_ROOT}/v0/episodes?subject_id={subject_id}&limit=100&offset=0",
                    priority=10, metadata={"selection_id": selection_id, "offset": 0},
                )
                self._enqueue(
                    db, key=f"subject-comments:{subject_id}:1", kind="subject_comments",
                    subject_id=subject_id,
                    url=f"{WEB_ROOT}/subject/{subject_id}/comments?page=1",
                    priority=20, metadata={"selection_id": selection_id, "page": 1},
                )
                self._enqueue(
                    db, key=f"review-list:{subject_id}:1", kind="review_list",
                    subject_id=subject_id,
                    url=f"{WEB_ROOT}/subject/{subject_id}/reviews?page=1",
                    priority=30, metadata={"selection_id": selection_id, "page": 1},
                )
        return self.status()

    def _wait(self) -> None:
        remaining = self.delay_seconds - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def _request(self, url: str) -> tuple[bytes, int, str, str]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._wait()
            self._last_request_at = time.monotonic()
            request = urllib.request.Request(url, headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json" if url.startswith(API_ROOT) else "text/html,application/xhtml+xml",
            })
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    content = response.read()
                    return content, int(response.status), response.geturl(), response.headers.get_content_type()
            except urllib.error.HTTPError as error:
                last_error = error
                if error.code not in {403, 429, 500, 502, 503, 504}:
                    raise
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 5 * 2 ** (attempt - 1)
                time.sleep(min(delay, 120))
            except (urllib.error.URLError, http.client.RemoteDisconnected,
                    http.client.IncompleteRead, TimeoutError) as error:
                last_error = error
                time.sleep(min(5 * 2 ** (attempt - 1), 120))
        raise RuntimeError(f"failed after {self.max_attempts} attempts: {url}") from last_error

    def _archive(self, content: bytes, *, content_type: str) -> tuple[str, str]:
        digest = hashlib.sha256(content).hexdigest()
        extension = ".json" if content_type == "application/json" else ".html"
        day = datetime.now(UTC).date().isoformat()
        target = self.layout.raw / "bgm_web" / day / f"{digest}{extension}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            fd, temp_name = tempfile.mkstemp(prefix=f".{digest}.", suffix=".tmp", dir=target.parent)
            temp = Path(temp_name)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, target)
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
        return f"sha256:{digest}", target.relative_to(self.layout.root).as_posix()

    def _expand(self, db: sqlite3.Connection, row: sqlite3.Row, content: bytes) -> None:
        kind = row["kind"]
        subject_id = int(row["subject_id"])
        metadata = json.loads(row["metadata_json"])
        if kind == "episode_meta":
            payload = json.loads(content)
            offset = int(metadata.get("offset", 0))
            for episode in payload.get("data", []):
                if int(episode.get("comment") or 0) <= 0:
                    continue
                episode_id = int(episode["id"])
                self._enqueue(
                    db, key=f"episode-comments:{episode_id}", kind="episode_comments",
                    subject_id=subject_id, object_id=episode_id,
                    url=f"{WEB_ROOT}/ep/{episode_id}", priority=70,
                    metadata={"episode_type": episode.get("type"), "episode_sort": episode.get("sort"),
                              "comment_count": episode.get("comment")},
                )
            total = int(payload.get("total") or 0)
            next_offset = offset + int(payload.get("limit") or 100)
            if next_offset < total:
                self._enqueue(
                    db, key=f"episode-meta:{subject_id}:{next_offset}", kind="episode_meta",
                    subject_id=subject_id,
                    url=f"{API_ROOT}/v0/episodes?subject_id={subject_id}&limit=100&offset={next_offset}",
                    priority=10, metadata={"offset": next_offset},
                )
            return
        if kind not in {"subject_comments", "review_list"}:
            return
        parser = BgmLinksParser()
        parser.feed(content.decode("utf-8", errors="replace"))
        current_page = int(metadata.get("page", 1))
        max_page = max(parser.page_numbers, default=current_page)
        if kind == "subject_comments" and current_page == 1:
            for page in sampled_pages(max_page, self.short_comment_page_cap):
                self._enqueue(
                    db, key=f"subject-comments:{subject_id}:{page}", kind=kind,
                    subject_id=subject_id,
                    url=f"{WEB_ROOT}/subject/{subject_id}/comments?page={page}", priority=40,
                    metadata={"page": page, "total_pages": max_page, "sampling": "quantile"},
                )
        if kind == "review_list":
            if current_page == 1:
                if max_page > 200:
                    raise RuntimeError(f"review listing exceeds safety cap: subject {subject_id}")
                for page in range(2, max_page + 1):
                    self._enqueue(
                        db, key=f"review-list:{subject_id}:{page}", kind=kind,
                        subject_id=subject_id,
                        url=f"{WEB_ROOT}/subject/{subject_id}/reviews/{page}.html", priority=50,
                        metadata={"page": page, "total_pages": max_page},
                    )
            for blog_id in parser.blog_ids:
                self._enqueue(
                    db, key=f"blog-review:{blog_id}", kind="blog_review",
                    subject_id=subject_id, object_id=blog_id,
                    url=f"{WEB_ROOT}/blog/{blog_id}", priority=60,
                )

    def reconcile_archived_review_pages(self) -> int:
        """Re-expand completed review page 1 snapshots after scheduler upgrades."""
        expanded = 0
        with closing(self.connect()) as db, db:
            rows = db.execute(
                "SELECT * FROM crawl_item WHERE kind='review_list' AND status='ok' "
                "AND json_extract(metadata_json,'$.page')=1 AND relative_path IS NOT NULL"
            ).fetchall()
            for row in rows:
                path = self.layout.root / row["relative_path"]
                if not path.is_file():
                    continue
                self._expand(db, row, path.read_bytes())
                expanded += 1
        return expanded

    def crawl(self, *, limit: int | None = None) -> dict[str, int]:
        attempted = 0
        with closing(self.connect()) as db, db:
            while limit is None or attempted < limit:
                row = db.execute(
                    "SELECT * FROM crawl_item WHERE status IN ('pending','failed') AND attempts < ? "
                    "ORDER BY priority, key LIMIT 1", (self.max_attempts,)
                ).fetchone()
                if row is None:
                    break
                attempted += 1
                db.execute("UPDATE crawl_item SET status='running', attempts=attempts+1, updated_at=? WHERE key=?",
                           (self._now(), row["key"]))
                db.commit()
                try:
                    content, status, final_url, content_type = self._request(row["url"])
                    digest, relative_path = self._archive(content, content_type=content_type)
                    self._expand(db, row, content)
                    db.execute(
                        "UPDATE crawl_item SET status='ok', content_hash=?, relative_path=?, http_status=?, "
                        "metadata_json=json_set(metadata_json,'$.final_url',?), last_error=NULL, updated_at=? WHERE key=?",
                        (digest, relative_path, status, final_url, self._now(), row["key"]),
                    )
                except urllib.error.HTTPError as error:
                    if error.code in {404, 410}:
                        db.execute(
                            "UPDATE crawl_item SET status='not_found', http_status=?, last_error=?, "
                            "updated_at=? WHERE key=?",
                            (error.code, f"HTTPError: {error}", self._now(), row["key"]),
                        )
                    else:
                        db.execute(
                            "UPDATE crawl_item SET status='failed', last_error=?, updated_at=? WHERE key=?",
                            (f"HTTPError: {error}", self._now(), row["key"]),
                        )
                except Exception as error:
                    db.execute("UPDATE crawl_item SET status='failed', last_error=?, updated_at=? WHERE key=?",
                               (f"{type(error).__name__}: {error}", self._now(), row["key"]))
                db.commit()
        result = self.status()
        result["attempted_this_run"] = attempted
        return result

    def crawl_parallel(self, *, workers: int = 3, limit: int | None = None) -> dict[str, int]:
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        lock = threading.Lock()
        attempted = 0
        stop = threading.Event()

        def fetch_one(db: sqlite3.Connection) -> sqlite3.Row | None:
            nonlocal attempted
            with lock:
                row = db.execute(
                    "SELECT * FROM crawl_item WHERE status IN ('pending','failed') AND attempts < ? "
                    "ORDER BY priority, key LIMIT 1", (self.max_attempts,)
                ).fetchone()
                if row is None:
                    return None
                attempted += 1
                db.execute(
                    "UPDATE crawl_item SET status='running', attempts=attempts+1, updated_at=? WHERE key=?",
                    (self._now(), row["key"])
                )
                db.commit()
            return row

        def worker() -> None:
            db = self.connect()
            last_req = 0.0
            try:
                while not stop.is_set():
                    if limit is not None and attempted >= limit:
                        break
                    row = fetch_one(db)
                    if row is None:
                        break
                    remaining = self.delay_seconds - (time.monotonic() - last_req)
                    if remaining > 0:
                        time.sleep(remaining)
                    last_req = time.monotonic()
                    try:
                        content, http_status, final_url, content_type = self._request(row["url"])
                        digest, relative_path = self._archive(content, content_type=content_type)
                        with lock:
                            db.execute(
                                "UPDATE crawl_item SET status='ok', content_hash=?, relative_path=?, http_status=?, "
                                "metadata_json=json_set(metadata_json,'$.final_url',?), last_error=NULL, updated_at=? "
                                "WHERE key=?",
                                (digest, relative_path, http_status, final_url, self._now(), row["key"]),
                            )
                            if row["kind"] in {"episode_meta", "subject_comments", "review_list"}:
                                self._expand(db, row, content)
                            db.commit()
                    except urllib.error.HTTPError as error:
                        with lock:
                            if error.code in {404, 410}:
                                db.execute(
                                    "UPDATE crawl_item SET status='not_found', http_status=?, last_error=?, updated_at=? WHERE key=?",
                                    (error.code, f"HTTPError: {error}", self._now(), row["key"]),
                                )
                            else:
                                db.execute(
                                    "UPDATE crawl_item SET status='failed', last_error=?, updated_at=? WHERE key=?",
                                    (f"HTTPError: {error}", self._now(), row["key"]),
                                )
                            db.commit()
                    except Exception as error:
                        with lock:
                            db.execute(
                                "UPDATE crawl_item SET status='failed', last_error=?, updated_at=? WHERE key=?",
                                (f"{type(error).__name__}: {error}", self._now(), row["key"]),
                            )
                            db.commit()
            finally:
                db.close()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker) for _ in range(workers)]
            for future in as_completed(futures):
                future.result()

        result = self.status()
        result["attempted_this_run"] = attempted
        return result

    def status(self) -> dict[str, Any]:
        with closing(self.connect()) as db:
            rows = db.execute("SELECT status, count(*) FROM crawl_item GROUP BY status").fetchall()
            kinds = db.execute("SELECT kind, count(*) FROM crawl_item GROUP BY kind").fetchall()
            matrix = db.execute(
                "SELECT kind, status, count(*) FROM crawl_item GROUP BY kind, status "
                "ORDER BY kind, status"
            ).fetchall()
        result = {f"status_{row[0]}": int(row[1]) for row in rows}
        result.update({f"kind_{row[0]}": int(row[1]) for row in kinds})
        result["total"] = sum(int(row[1]) for row in rows)
        by_kind: dict[str, dict[str, int]] = {}
        for kind, status, count in matrix:
            by_kind.setdefault(str(kind), {})[str(status)] = int(count)
        result["by_kind"] = by_kind
        unfinished = sum(
            int(row[1]) for row in rows if row[0] not in {"ok", "not_found"}
        )
        result["unfinished"] = unfinished
        result["minimum_remaining_seconds"] = int(unfinished * self.delay_seconds)
        return result

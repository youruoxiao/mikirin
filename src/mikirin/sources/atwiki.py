from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

from mikirin.atomic import atomic_write_json
from mikirin.layout import ProjectLayout


ATWIKI_ROOT = "https://w.atwiki.jp/sakuga/"
ATWIKI_LIST = urllib.parse.urljoin(ATWIKI_ROOT, "list")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/138.0 Safari/537.36 MikirinArchive/0.1"
)
_PAGE_RE = re.compile(r"^/sakuga/pages/(\d+)\.html$")


class _PageLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pages: dict[int, str] = {}
        self._active_id: int | None = None
        self._active_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href") or ""
        parsed = urllib.parse.urlparse(urllib.parse.urljoin(ATWIKI_ROOT, href))
        if parsed.hostname not in {"w.atwiki.jp", "www18.atwiki.jp"}:
            return
        match = _PAGE_RE.fullmatch(parsed.path)
        if match:
            self._active_id = int(match.group(1))
            self._active_text = []

    def handle_data(self, data: str) -> None:
        if self._active_id is not None:
            self._active_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_id is not None:
            title = " ".join("".join(self._active_text).split())
            self.pages[self._active_id] = title
            self._active_id = None
            self._active_text = []


@dataclass(frozen=True, slots=True)
class AtwikiPage:
    page_id: int
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class FetchRecord:
    key: str
    url: str
    final_url: str
    title: str
    status: int
    content_hash: str
    byte_size: int
    relative_path: str
    fetched_at: str


class AtwikiCrawler:
    def __init__(
        self,
        layout: ProjectLayout,
        *,
        delay_seconds: float = 2.0,
        timeout_seconds: float = 30.0,
        max_attempts: int = 5,
    ) -> None:
        if delay_seconds < 1.0:
            raise ValueError("作画 Wiki 抓取间隔不得低于 1 秒")
        self.layout = layout
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self._last_request_at = 0.0

    @property
    def queue_path(self) -> Path:
        return self.layout.manifests / "atwiki" / "page-queue.json"

    @property
    def state_path(self) -> Path:
        return self.layout.manifests / "atwiki" / "crawl-state.json"

    def _wait_for_rate_limit(self) -> None:
        remaining = self.delay_seconds - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def _request(self, url: str) -> tuple[bytes, int, str]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._wait_for_rate_limit()
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    self._last_request_at = time.monotonic()
                    return response.read(), int(response.status), response.geturl()
            except urllib.error.HTTPError as error:
                self._last_request_at = time.monotonic()
                last_error = error
                if error.code not in {403, 429, 500, 502, 503, 504}:
                    raise
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 5 * 2 ** (attempt - 1)
                time.sleep(min(delay, 120))
            except (
                urllib.error.URLError,
                http.client.RemoteDisconnected,
                http.client.IncompleteRead,
                TimeoutError,
            ) as error:
                self._last_request_at = time.monotonic()
                last_error = error
                time.sleep(min(5 * 2 ** (attempt - 1), 120))
        raise RuntimeError(f"连续 {self.max_attempts} 次请求失败: {url}") from last_error

    def _archive_bytes(
        self,
        *,
        key: str,
        url: str,
        final_url: str,
        title: str,
        status: int,
        content: bytes,
    ) -> FetchRecord:
        digest = hashlib.sha256(content).hexdigest()
        day = datetime.now(UTC).date().isoformat()
        destination = self.layout.raw / "atwiki" / day / f"{digest}.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{digest}.", suffix=".tmp", dir=destination.parent
            )
            temp_path = Path(temp_name)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, destination)
            except BaseException:
                temp_path.unlink(missing_ok=True)
                raise
        record = FetchRecord(
            key=key,
            url=url,
            final_url=final_url,
            title=title,
            status=status,
            content_hash=f"sha256:{digest}",
            byte_size=len(content),
            relative_path=destination.relative_to(self.layout.root).as_posix(),
            fetched_at=datetime.now(UTC).isoformat(),
        )
        metadata_path = self.layout.manifests / "atwiki" / "assets" / f"{key}.json"
        atomic_write_json(metadata_path, asdict(record))
        return record

    def discover(self) -> tuple[AtwikiPage, ...]:
        self.layout.initialize()
        discovered: dict[int, AtwikiPage] = {}
        page_number = 1
        while True:
            url = ATWIKI_LIST if page_number == 1 else f"{ATWIKI_LIST}?pp={page_number}"
            content, status, final_url = self._request(url)
            self._archive_bytes(
                key=f"list-{page_number}",
                url=url,
                final_url=final_url,
                title=f"页面列表 {page_number}",
                status=status,
                content=content,
            )
            parser = _PageLinkParser()
            parser.feed(content.decode("utf-8", errors="replace"))
            if not parser.pages:
                break
            new_count = 0
            for page_id, title in parser.pages.items():
                if page_id not in discovered:
                    new_count += 1
                discovered[page_id] = AtwikiPage(
                    page_id=page_id,
                    title=title,
                    url=f"https://w.atwiki.jp/sakuga/pages/{page_id}.html",
                )
            if new_count == 0 or len(parser.pages) < 100:
                break
            page_number += 1
            if page_number > 100:
                raise RuntimeError("作画 Wiki 列表分页超过安全上限 100")
        queue = tuple(sorted(discovered.values(), key=lambda page: page.page_id))
        atomic_write_json(
            self.queue_path,
            {
                "discovered_at": datetime.now(UTC).isoformat(),
                "count": len(queue),
                "pages": [asdict(page) for page in queue],
            },
        )
        return queue

    def load_queue(self) -> tuple[AtwikiPage, ...]:
        with self.queue_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return tuple(AtwikiPage(**page) for page in payload["pages"])

    def crawl(self, pages: tuple[AtwikiPage, ...], *, limit: int | None = None) -> dict[str, int]:
        state: dict[str, dict[str, object]] = {}
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle).get("pages", {})
        attempted = 0
        fetched = 0
        skipped = 0
        failed = 0
        for page in pages:
            key = str(page.page_id)
            if state.get(key, {}).get("status") == "ok":
                skipped += 1
                continue
            if limit is not None and attempted >= limit:
                break
            attempted += 1
            try:
                content, status, final_url = self._request(page.url)
                record = self._archive_bytes(
                    key=f"page-{page.page_id}",
                    url=page.url,
                    final_url=final_url,
                    title=page.title,
                    status=status,
                    content=content,
                )
                state[key] = {"status": "ok", "record": asdict(record)}
                fetched += 1
            except Exception as error:
                state[key] = {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "updated_at": datetime.now(UTC).isoformat(),
                }
                failed += 1
            atomic_write_json(
                self.state_path,
                {"updated_at": datetime.now(UTC).isoformat(), "pages": state},
            )
        return {"attempted": attempted, "fetched": fetched, "skipped": skipped, "failed": failed}

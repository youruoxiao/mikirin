from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
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


DEFAULT_ROOT = "https://bangumi.tv"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/126.0 Safari/537.36 MikirinArchive/0.1"
)


@dataclass(frozen=True, slots=True)
class ListedSubject:
    subject_id: int
    title: str


class SubjectListParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.subjects: list[ListedSubject] = []
        self.page_numbers: set[int] = set()
        self._active_id: int | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        values = dict(attrs)
        href = values.get("href") or ""
        query = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        for value in query.get("page", []):
            if value.isdigit():
                self.page_numbers.add(int(value))
        classes = set((values.get("class") or "").split())
        if "l" not in classes:
            return
        parsed = urllib.parse.urlparse(href)
        if parsed.netloc:
            return
        path = parsed.path
        parts = path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "subject" and parts[1].isdigit():
            self._active_id = int(parts[1])
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._active_id is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._active_id is not None:
            title = " ".join("".join(self._text).split())
            self.subjects.append(ListedSubject(self._active_id, title))
            self._active_id = None
            self._text = []


class BgmSampleDiscoverer:
    def __init__(
        self,
        layout: ProjectLayout,
        *,
        root_url: str = DEFAULT_ROOT,
        delay_seconds: float = 1.0,
        timeout_seconds: float = 30.0,
        max_attempts: int = 5,
    ) -> None:
        if delay_seconds < 1.0:
            raise ValueError("Bangumi webpage requests must be at least one second apart")
        self.layout = layout
        self.root_url = root_url.rstrip("/")
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self._last_request_at = 0.0

    def _wait(self) -> None:
        remaining = self.delay_seconds - (time.monotonic() - self._last_request_at)
        if remaining > 0:
            time.sleep(remaining)

    def _request(self, url: str) -> tuple[bytes, int, str]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._wait()
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    content = response.read()
                    self._last_request_at = time.monotonic()
                    return content, int(response.status), response.geturl()
            except urllib.error.HTTPError as error:
                self._last_request_at = time.monotonic()
                last_error = error
                if error.code not in {403, 429, 500, 502, 503, 504}:
                    raise
                retry_after = error.headers.get("Retry-After")
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else 5 * 2 ** (attempt - 1)
                )
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
        raise RuntimeError(f"failed after {self.max_attempts} attempts: {url}") from last_error

    def _archive(self, content: bytes, *, key: str, url: str, final_url: str) -> dict[str, object]:
        digest = hashlib.sha256(content).hexdigest()
        day = datetime.now(UTC).date().isoformat()
        target = self.layout.raw / "bgm_web" / day / f"{digest}.html"
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
        return {
            "key": key,
            "url": url,
            "final_url": final_url,
            "content_hash": f"sha256:{digest}",
            "byte_size": len(content),
            "relative_path": target.relative_to(self.layout.root).as_posix(),
            "fetched_at": datetime.now(UTC).isoformat(),
        }

    def _fetch_list(
        self, url: str, *, key: str
    ) -> tuple[list[ListedSubject], dict[str, object], int]:
        content, status, final_url = self._request(url)
        if status != 200:
            raise RuntimeError(f"unexpected HTTP status {status}: {url}")
        parser = SubjectListParser()
        parser.feed(content.decode("utf-8", errors="replace"))
        asset = self._archive(content, key=key, url=url, final_url=final_url)
        return parser.subjects, asset, max(parser.page_numbers, default=1)

    def discover(
        self,
        *,
        start_year: int,
        end_year: int,
        yearly_limit: int,
        rank_limit: int,
        user_ids: list[str],
        user_page_cap: int = 200,
    ) -> Path:
        if start_year > end_year or yearly_limit < 1 or rank_limit < 1:
            raise ValueError("invalid selection limits")
        self.layout.initialize()
        selected: dict[int, dict[str, object]] = {}
        assets: list[dict[str, object]] = []

        def add(subject: ListedSubject, source: dict[str, object]) -> None:
            record = selected.setdefault(
                subject.subject_id,
                {"subject_id": subject.subject_id, "title": subject.title, "sources": []},
            )
            if not record["title"] and subject.title:
                record["title"] = subject.title
            record["sources"].append(source)  # type: ignore[union-attr]

        for year in range(start_year, end_year + 1):
            remaining = yearly_limit
            for page in range(1, math.ceil(yearly_limit / 24) + 1):
                base = f"{self.root_url}/anime/browser/airtime/{year}?sort=trends"
                url = base if page == 1 else f"{base}&page={page}"
                subjects, asset, _ = self._fetch_list(url, key=f"year-{year}-page-{page}")
                assets.append(asset)
                for offset, subject in enumerate(subjects[:remaining]):
                    add(subject, {"kind": "yearly_trends", "year": year, "rank": (page - 1) * 24 + offset + 1})
                remaining -= min(remaining, len(subjects))
                if remaining <= 0:
                    break

        remaining = rank_limit
        for page in range(1, math.ceil(rank_limit / 24) + 1):
            base = f"{self.root_url}/anime/browser?sort=rank"
            url = base if page == 1 else f"{base}&page={page}"
            subjects, asset, _ = self._fetch_list(url, key=f"score-rank-page-{page}")
            assets.append(asset)
            for offset, subject in enumerate(subjects[:remaining]):
                add(subject, {"kind": "score_rank", "rank": (page - 1) * 24 + offset + 1})
            remaining -= min(remaining, len(subjects))
            if remaining <= 0:
                break

        for user_id in user_ids:
            if not user_id.isdigit():
                raise ValueError(f"Bangumi user ID must be numeric: {user_id!r}")
            discovered_last_page = 1
            for page in range(1, user_page_cap + 1):
                base = f"{self.root_url}/anime/list/{user_id}/collect"
                url = base if page == 1 else f"{base}?page={page}"
                subjects, asset, max_page = self._fetch_list(
                    url, key=f"user-{user_id}-page-{page}"
                )
                assets.append(asset)
                discovered_last_page = max(discovered_last_page, max_page)
                for offset, subject in enumerate(subjects):
                    add(subject, {"kind": "user_collect", "user_id": user_id, "position": (page - 1) * 24 + offset + 1})
                if page >= discovered_last_page:
                    break
            else:
                raise RuntimeError(f"user collection exceeded page cap {user_page_cap}: {user_id}")

        generated_at = datetime.now(UTC)
        selection_id = generated_at.strftime("selection-%Y%m%dT%H%M%SZ")
        output = self.layout.manifests / "bgm_web" / f"{selection_id}.json"
        atomic_write_json(output, {
            "schema_version": 1,
            "selection_id": selection_id,
            "generated_at": generated_at.isoformat(),
            "root_url": self.root_url,
            "policy": {
                "start_year": start_year, "end_year": end_year,
                "yearly_limit": yearly_limit, "rank_limit": rank_limit,
                "user_ids": user_ids,
            },
            "asset_count": len(assets),
            "assets": assets,
            "subject_count": len(selected),
            "subjects": sorted(selected.values(), key=lambda item: int(item["subject_id"])),
        })
        atomic_write_json(self.layout.manifests / "bgm_web" / "current-selection.json", {
            "selection_id": selection_id,
            "path": output.relative_to(self.layout.root).as_posix(),
        })
        return output

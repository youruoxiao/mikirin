"""Opt-in RSS/Atom topics, stored independently alongside UserStore's tables.

CLI/TUI integration (no scheduler or startup hooks are installed here)::

    service = ProactiveService(layout.user_db, fetcher=fetch_http_feed)
    service.subscribe("bgm:subject:123", user_supplied_rss_url)
    service.configure(enabled=True, probability=0.30)
    service.poll()  # explicit refresh; never discovers feeds from watch_progress
    service.deliver(lambda topic: print(topic.content))  # once at next startup

All public records are dataclasses suitable for dataclasses.asdict(). A callback
returning None/True acknowledges delivery; False or an exception releases it for
retry. Call deliver once per startup, not once per UI refresh. For asynchronous
UIs, claim() applies the same probability gate and returns a lease; acknowledge
with ack(id, lease_token), or release on failure. Finish within the lease (60s by
default), otherwise another process may retry. clock is Unix seconds, rng is a
zero-argument callable returning a value in [0, 1).

This is an at-least-once retry queue, NOT exactly-once external display. A crash
after display but before ack, or display after a lease expires, can cause a second
display. Persist/deduplicate topic.id at the display sink if that is unacceptable.
The SQLite acknowledgment itself is atomic, fenced by an unexpired lease token.
Disabling stops new work; it cannot recall an already-running HTTP call/callback.
Acknowledging already-displayed work remains permitted while disabled.

The default fetcher is None: construction, subscription and delivery never access
the network. A trusted injected fetcher must honor timeout/max_bytes and return
bytes; it must read only the supplied HTTP(S) feed URL, without redirects or
secondary requests. The supplied fetch_http_feed implements this policy, with
socket timeouts and an elapsed-time budget (OS DNS resolution is not cancellable
by urllib). No model, enclosure, article, torrent or media is ever fetched.
Feed metadata is untrusted data, never an instruction or an automatic open target.
Generated content deliberately omits item titles/summaries to avoid plot spoilers.
"""

from __future__ import annotations

import hashlib
import math
import random
import sqlite3
import time
import unicodedata
import uuid
from collections.abc import Callable
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator, Protocol
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from xml.etree import ElementTree as ET


MAX_FEED_BYTES = 1_048_576
FETCH_TIMEOUT = 10.0
MAX_ITEMS = 500

PROACTIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS proactive_settings(
    id INTEGER PRIMARY KEY CHECK(id=1),
    enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0,1)),
    probability REAL NOT NULL DEFAULT 0.30 CHECK(probability BETWEEN 0 AND 1)
);
INSERT OR IGNORE INTO proactive_settings(id) VALUES(1);
CREATE TABLE IF NOT EXISTS proactive_subscription(
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    url TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)),
    created_at TEXT NOT NULL,
    UNIQUE(subject_id, url)
);
CREATE TABLE IF NOT EXISTS proactive_topic(
    id TEXT PRIMARY KEY,
    subscription_id TEXT NOT NULL REFERENCES proactive_subscription(id),
    subject_id TEXT NOT NULL,
    item_key TEXT NOT NULL,
    title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    read_at TEXT,
    cancelled_at TEXT,
    lease_token TEXT,
    lease_until REAL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    UNIQUE(subscription_id, item_key)
);
CREATE INDEX IF NOT EXISTS proactive_pending
    ON proactive_topic(read_at, cancelled_at, lease_until, created_at);
"""


class FeedFetcher(Protocol):
    def __call__(self, url: str, *, timeout: float, max_bytes: int) -> bytes: ...


@dataclass(frozen=True, slots=True)
class Subscription:
    id: str
    subject_id: str
    url: str
    created_at: str


@dataclass(frozen=True, slots=True)
class ProactiveStatus:
    enabled: bool
    probability: float
    subscriptions: int
    unread: int
    read: int
    leased: int
    cancelled: int


@dataclass(frozen=True, slots=True)
class ProactiveTopic:
    id: str
    subscription_id: str
    subject_id: str
    item_key: str
    title: str
    source_url: str
    content: str
    created_at: str


@dataclass(frozen=True, slots=True)
class ClaimedTopic(ProactiveTopic):
    lease_token: str
    lease_until: float
    attempts: int


@dataclass(slots=True)
class PollOutcome:
    polled: int = 0
    enqueued: int = 0
    duplicates: int = 0
    errors: dict[str, str] = field(default_factory=dict)  # subscription id -> error
    disabled: bool = False


@dataclass(slots=True)
class DeliveryOutcome:
    delivered: list[ProactiveTopic] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # topic id -> error
    skipped: str | None = None  # disabled / empty / probability


def _http_url(url: str) -> str:
    if not isinstance(url, str) or not url or len(url) > 4096:
        raise ValueError("feed URL must be a nonempty HTTP(S) URL, at most 4096 characters")
    if any(c.isspace() or unicodedata.category(c).startswith("C") for c in url):
        raise ValueError("feed URL cannot contain whitespace or control characters")
    parts = urlsplit(url)
    if (parts.scheme.lower() not in {"http", "https"} or not parts.hostname
            or parts.username is not None or parts.password is not None or parts.fragment):
        raise ValueError("only explicit HTTP(S) URLs without credentials/fragments are allowed")
    if parts.port is not None and not 1 <= parts.port <= 65535:
        raise ValueError("invalid feed URL port")
    # Preserve query and path, which may be signed/case-sensitive.
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                       parts.path or "/", parts.query, ""))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("feed redirects are disabled; subscribe to the final URL explicitly")


def fetch_http_feed(url: str, *, timeout: float, max_bytes: int) -> bytes:
    """Read one explicit feed, without redirects, proxies or decompression.

    timeout bounds socket waits and is checked between read1 operations; a slow
    in-flight read can add at most one socket timeout to the elapsed budget,
    excluding OS DNS resolution. Custom hard-deadline transports can be injected.
    """
    url = _http_url(url)
    if not math.isfinite(timeout) or timeout <= 0 or max_bytes < 1:
        raise ValueError("positive finite timeout and size limit required")
    started = time.monotonic()
    request = Request(url, headers={"Accept": "application/rss+xml, application/atom+xml, "
                                   "application/xml, text/xml",
                                   "Accept-Encoding": "identity",
                                   "User-Agent": "Mikirin-RSS/1.0"})
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        if response.status != 200 or response.geturl() != url:
            raise ValueError("unexpected feed response")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("compressed feeds are not accepted")
        content_length = response.headers.get("Content-Length")
        if content_length is not None and not 0 <= int(content_length) <= max_bytes:
            raise ValueError("feed exceeds size limit")
        data = bytearray()
        while True:
            if time.monotonic() - started >= timeout:
                raise TimeoutError("feed read budget expired")
            chunk = response.read1(min(65536, max_bytes + 1 - len(data)))
            if time.monotonic() - started >= timeout:
                raise TimeoutError("feed read budget expired")
            data.extend(chunk)
            if len(data) > max_bytes:
                raise ValueError("feed exceeds size limit")
            if not chunk:
                return bytes(data)


class _SafeTree(ET.TreeBuilder):
    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.nodes = 0

    def doctype(self, name, pubid, system):
        # Parser callback, not a byte search: also rejects UTF-16/32 DTDs.
        raise ValueError("DTD/entity declarations are forbidden")

    def start(self, tag, attrs):
        self.depth += 1
        self.nodes += 1
        if self.depth > 32 or self.nodes > 20000:
            raise ValueError("feed XML complexity limit exceeded")
        return super().start(tag, attrs)

    def end(self, tag):
        element = super().end(tag)
        self.depth -= 1
        return element


def _text(element: ET.Element, name: str) -> str:
    child = element.find(name)
    return "" if child is None else "".join(child.itertext()).strip()


def _display_text(value: str, limit: int = 500) -> str:
    return " ".join("".join(
        c if not unicodedata.category(c).startswith("C") else " " for c in value
    ).split())[:limit]


def _feed_items(data: bytes, feed_url: str) -> list[tuple[str, str, str]]:
    if not isinstance(data, bytes) or len(data) > MAX_FEED_BYTES:
        raise ValueError("fetcher must return bytes within the feed size limit")
    root = ET.fromstring(data, parser=ET.XMLParser(target=_SafeTree()))
    atom = "{http://www.w3.org/2005/Atom}"
    rdf = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
    rss1 = "{http://purl.org/rss/1.0/}"
    if root.tag == "rss":
        entries, prefix = root.findall("channel/item"), ""
    elif root.tag == atom + "feed":
        entries, prefix = root.findall(atom + "entry"), atom
    elif root.tag == rdf + "RDF":
        entries, prefix = root.findall(rss1 + "item"), rss1
    else:
        raise ValueError("expected an RSS or Atom feed")
    if len(entries) > MAX_ITEMS:
        raise ValueError("feed item limit exceeded")
    result = []
    for entry in entries:
        title = _text(entry, prefix + "title")
        identity = _text(entry, prefix + ("id" if prefix == atom else "guid"))
        if prefix == rss1:
            identity = identity or entry.get(rdf + "about", "")
        link = ""
        if prefix == atom:
            for candidate in entry.findall(atom + "link"):
                if candidate.get("rel", "alternate") == "alternate":
                    link = candidate.get("href", "")
                    break
        else:
            link = _text(entry, prefix + "link")
        source = feed_url
        if link:
            try:
                source = _http_url(urljoin(feed_url, link))
            except ValueError:
                link = ""  # Never retain non-HTTP links as UI targets.
        if identity:
            key = "id:" + identity
        elif link:
            key = "link:" + source
        elif title:
            # Stable fallback: title only. Ambiguous equal titles intentionally
            # collapse; mutable timestamps/descriptions must not duplicate items.
            key = "title:" + " ".join(title.split())
        else:
            continue
        result.append((hashlib.sha256(key.encode()).hexdigest(),
                       _display_text(title), source))
    return result


class ProactiveService:
    """Independent SQLite tables; never initializes/rebuilds UserStore or FTS."""

    def __init__(
        self, user_db: str | Path, *, fetcher: FeedFetcher | None = None,
        rng: Callable[[], float] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.path = Path(user_db)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fetcher = fetcher
        self.rng = rng if rng is not None else random.random
        self.clock = clock if clock is not None else time.time
        with self._db() as db:
            db.executescript(PROACTIVE_SCHEMA)

    @contextmanager
    def _db(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=5)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db

    def _now(self) -> str:
        return datetime.fromtimestamp(self.clock(), UTC).isoformat()

    def configure(
        self, *, enabled: bool | None = None, probability: float | None = None,
    ) -> ProactiveStatus:
        if enabled is not None and not isinstance(enabled, bool):
            raise ValueError("enabled must be a bool")
        if probability is not None and (
            isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or not 0 <= probability <= 1
        ):
            raise ValueError("probability must be finite and between 0 and 1")
        with self._db(write=True) as db:
            if enabled is not None:
                db.execute("UPDATE proactive_settings SET enabled=? WHERE id=1", (int(enabled),))
            if probability is not None:
                db.execute("UPDATE proactive_settings SET probability=? WHERE id=1", (probability,))
        return self.status()

    def subscribe(self, subject_id: str, url: str) -> Subscription:
        if (not isinstance(subject_id, str) or not subject_id.strip() or len(subject_id) > 512
                or any(unicodedata.category(c).startswith("C") for c in subject_id)):
            raise ValueError("subject_id must be a nonempty, control-free identifier")
        subject_id, url = subject_id.strip(), _http_url(url)
        with self._db(write=True) as db:
            db.execute(
                "INSERT INTO proactive_subscription(id,subject_id,url,created_at) VALUES(?,?,?,?) "
                "ON CONFLICT(subject_id,url) DO UPDATE SET active=1",
                (uuid.uuid4().hex, subject_id, url, self._now()),
            )
            row = db.execute(
                "SELECT id,subject_id,url,created_at FROM proactive_subscription "
                "WHERE subject_id=? AND url=?", (subject_id, url),
            ).fetchone()
        return Subscription(**dict(row))

    def unsubscribe(self, subscription_id: str) -> bool:
        """Cancel pending work, retaining identity tombstones across re-subscription."""
        with self._db(write=True) as db:
            changed = db.execute(
                "UPDATE proactive_subscription SET active=0 WHERE id=? AND active=1",
                (subscription_id,),
            ).rowcount
            db.execute(
                "UPDATE proactive_topic SET cancelled_at=?,lease_token=NULL,lease_until=NULL "
                "WHERE subscription_id=? AND read_at IS NULL AND cancelled_at IS NULL",
                (self._now(), subscription_id),
            )
        return bool(changed)

    def subscriptions(self) -> list[Subscription]:
        with self._db() as db:
            rows = db.execute(
                "SELECT id,subject_id,url,created_at FROM proactive_subscription "
                "WHERE active=1 ORDER BY created_at,rowid"
            ).fetchall()
        return [Subscription(**dict(row)) for row in rows]

    def status(self) -> ProactiveStatus:
        with self._db() as db:
            # One statement: settings and counts are a consistent SQLite snapshot.
            row = db.execute(
                """SELECT enabled,probability,
                (SELECT count(*) FROM proactive_subscription WHERE active=1),
                (SELECT count(*) FROM proactive_topic WHERE read_at IS NULL
                    AND cancelled_at IS NULL),
                (SELECT count(*) FROM proactive_topic WHERE read_at IS NOT NULL),
                (SELECT count(*) FROM proactive_topic WHERE read_at IS NULL
                    AND cancelled_at IS NULL AND lease_until>?),
                (SELECT count(*) FROM proactive_topic WHERE cancelled_at IS NOT NULL)
                FROM proactive_settings WHERE id=1""", (self.clock(),),
            ).fetchone()
        return ProactiveStatus(bool(row[0]), *tuple(row)[1:])

    @staticmethod
    def _enabled(db: sqlite3.Connection) -> bool:
        return bool(db.execute("SELECT enabled FROM proactive_settings WHERE id=1").fetchone()[0])

    def poll(self) -> PollOutcome:
        outcome = PollOutcome()
        if not self.status().enabled:
            outcome.disabled = True
            return outcome
        for sub in self.subscriptions():
            # Recheck between subscriptions; do not hold a DB lock across I/O.
            with self._db() as db:
                if not self._enabled(db):
                    outcome.disabled = True
                    break
                active = db.execute(
                    "SELECT active FROM proactive_subscription WHERE id=?", (sub.id,),
                ).fetchone()[0]
            if not active:
                continue
            if self.fetcher is None:
                outcome.errors[sub.id] = "no feed fetcher configured"
                continue
            try:
                outcome.polled += 1
                started = time.monotonic()
                data = self.fetcher(sub.url, timeout=FETCH_TIMEOUT, max_bytes=MAX_FEED_BYTES)
                if time.monotonic() - started >= FETCH_TIMEOUT:
                    raise TimeoutError("feed fetch budget expired")
                items = _feed_items(data, sub.url)
                if time.monotonic() - started >= FETCH_TIMEOUT:
                    raise TimeoutError("feed parse budget expired")
                enqueued = 0
                with self._db(write=True) as db:
                    if not self._enabled(db):
                        outcome.disabled = True
                        break
                    if not db.execute(
                        "SELECT active FROM proactive_subscription WHERE id=?", (sub.id,),
                    ).fetchone()[0]:
                        continue
                    for key, title, source in items:
                        # Generate only after checking consent under the write lock.
                        topic_id = hashlib.sha256(f"{sub.id}:{key}".encode()).hexdigest()
                        inserted = db.execute(
                            "INSERT OR IGNORE INTO proactive_topic"
                            "(id,subscription_id,subject_id,item_key,title,source_url,content,"
                            "created_at) VALUES(?,?,?,?,?,?,?,?)",
                            (topic_id, sub.id, sub.subject_id, key, title, source,
                             f"你订阅的 {sub.subject_id} 有新更新。想聊聊最近看到的部分吗？",
                             self._now()),
                        ).rowcount
                        enqueued += inserted
                # Report only committed work, including when a transaction fails
                # after inserting some (but not all) items in this feed.
                outcome.enqueued += enqueued
                outcome.duplicates += len(items) - enqueued
            except Exception as error:
                # Transport errors may contain secret feed query strings.
                outcome.errors[sub.id] = type(error).__name__
        return outcome

    def claim(self, *, lease_seconds: float = 60) -> ClaimedTopic | None:
        """Claim at most one topic with a probability gate; does not mark it read."""
        return self._claim(lease_seconds=lease_seconds, gate=True)[0]

    def _claim(self, *, lease_seconds: float, gate: bool) -> tuple[ClaimedTopic | None, str | None]:
        if not math.isfinite(lease_seconds) or not 0 < lease_seconds <= 3600:
            raise ValueError("lease_seconds must be finite and in (0, 3600]")
        with self._db(write=True) as db:
            settings = db.execute("SELECT * FROM proactive_settings WHERE id=1").fetchone()
            if not settings["enabled"]:
                return None, "disabled"
            now = self.clock()
            row = db.execute(
                "SELECT t.* FROM proactive_topic t JOIN proactive_subscription s "
                "ON s.id=t.subscription_id WHERE s.active=1 AND t.read_at IS NULL "
                "AND t.cancelled_at IS NULL AND (t.lease_until IS NULL OR t.lease_until<=?) "
                "ORDER BY t.created_at,t.rowid LIMIT 1", (now,),
            ).fetchone()
            if row is None:
                return None, "empty"
            probability = settings["probability"]
            if probability == 0:
                return None, "probability"
            if gate and probability < 1:
                draw = self.rng()
                if not math.isfinite(draw) or not 0 <= draw < 1:
                    raise ValueError("rng must return a finite number in [0, 1)")
                if draw >= probability:
                    return None, "probability"
            token = uuid.uuid4().hex
            until = now + lease_seconds
            db.execute(
                "UPDATE proactive_topic SET lease_token=?,lease_until=?,attempts=attempts+1 "
                "WHERE id=?", (token, until, row["id"]),
            )
            values = {name: row[name] for name in ProactiveTopic.__dataclass_fields__}
            return ClaimedTopic(**values, lease_token=token, lease_until=until,
                                attempts=row["attempts"] + 1), None

    def ack(self, topic_id: str, lease_token: str) -> bool:
        """Atomic unread -> read transition; stale/expired tokens cannot acknowledge."""
        with self._db(write=True) as db:
            return bool(db.execute(
                "UPDATE proactive_topic SET read_at=?,lease_token=NULL,lease_until=NULL,"
                "last_error='' WHERE id=? AND lease_token=? AND lease_until>? "
                "AND read_at IS NULL AND cancelled_at IS NULL",
                (self._now(), topic_id, lease_token, self.clock()),
            ).rowcount)

    def release(self, topic_id: str, lease_token: str, *, error: str = "") -> bool:
        with self._db(write=True) as db:
            return bool(db.execute(
                "UPDATE proactive_topic SET lease_token=NULL,lease_until=NULL,last_error=? "
                "WHERE id=? AND lease_token=? AND lease_until>? AND read_at IS NULL "
                "AND cancelled_at IS NULL",
                (_display_text(error), topic_id, lease_token, self.clock()),
            ).rowcount)

    def deliver(
        self, callback: Callable[[ClaimedTopic], bool | None], *, limit: int = 1,
    ) -> DeliveryOutcome:
        """One probability draw per call; stop on first failure to avoid a retry loop.

        No database lock is held while calling UI code. KeyboardInterrupt and
        other BaseExceptions propagate after releasing the current claim.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        outcome = DeliveryOutcome()
        for index in range(limit):
            topic, reason = self._claim(lease_seconds=60, gate=index == 0)
            if topic is None:
                outcome.skipped = reason
                break
            try:
                # Changes during a completed callback cannot undo display, but
                # no further callback should start after observed disable/cancel.
                with self._db() as db:
                    allowed = self._enabled(db) and db.execute(
                        "SELECT 1 FROM proactive_topic WHERE id=? AND lease_token=? "
                        "AND lease_until>? AND cancelled_at IS NULL AND read_at IS NULL",
                        (topic.id, topic.lease_token, self.clock()),
                    ).fetchone() is not None
                if not allowed:
                    self.release(topic.id, topic.lease_token)
                    outcome.skipped = "disabled" if not self.status().enabled else "empty"
                    break
                response = callback(topic)
                if response is False:
                    self.release(topic.id, topic.lease_token, error="callback returned False")
                    outcome.failed[topic.id] = "callback returned False"
                    break
                if response is not None and response is not True:
                    raise TypeError("delivery callback must return None, True or False")
            except BaseException as error:
                self.release(topic.id, topic.lease_token, error=type(error).__name__)
                if not isinstance(error, Exception):
                    raise
                outcome.failed[topic.id] = type(error).__name__
                break
            if not self.ack(topic.id, topic.lease_token):
                outcome.failed[topic.id] = "lease lost after callback; display may have occurred"
                break
            outcome.delivered.append(topic)
        return outcome

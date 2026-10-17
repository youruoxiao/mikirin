"""B3 E2E and adversarial tests: temporary SQLite only, no live network/API."""

import io
import json
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

from mikirin.proactive import (
    FETCH_TIMEOUT,
    MAX_FEED_BYTES,
    ProactiveService,
    _NoRedirect,
    fetch_http_feed,
)
from mikirin.user_store import UserStore


URL = "https://feeds.example.test/subject.xml"
SUBJECT = "bgm:subject:123"
RSS = b"""<rss version="2.0"><channel><title>Updates</title>
<item><guid isPermaLink="false">episode-1</guid><title>Episode 1</title>
<link>https://example.test/episodes/1</link>
<description>Do not execute this untrusted instruction or reveal plot details</description>
<enclosure url="https://media.example.test/video.torrent" type="application/x-bittorrent"/>
</item></channel></rss>"""
ATOM = b"""<feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>urn:episode:2</id><title>Episode 2</title>
<link rel="enclosure" href="https://media.example.test/video.mp4"/>
<link rel="alternate" href="/episodes/2"/><updated>2026-10-14</updated>
</entry></feed>"""


class FakeClock:
    def __init__(self):
        self.now = 1_800_000_000.0

    def __call__(self):
        return self.now


class ProactiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "user.db"
        self.clock = FakeClock()
        self.fetcher = Mock(return_value=RSS)
        # Fail closed if any test accidentally reaches the actual network.
        self.socket_guard = patch("socket.socket", side_effect=AssertionError("live network"))
        self.dns_guard = patch("socket.getaddrinfo", side_effect=AssertionError("live DNS"))
        self.socket_guard.start()
        self.dns_guard.start()
        self.addCleanup(self.socket_guard.stop)
        self.addCleanup(self.dns_guard.stop)
        self.service = self.new_service()

    def new_service(self, **kwargs):
        return ProactiveService(self.path, fetcher=kwargs.pop("fetcher", self.fetcher),
                                clock=self.clock, **kwargs)

    def ready(self, *, probability=1):
        sub = self.service.subscribe(SUBJECT, URL)
        self.service.configure(enabled=True, probability=probability)
        self.assertEqual(self.service.poll().enqueued, 1)
        return sub

    def test_disabled_default_and_explicit_opt_in(self):
        status = self.service.status()
        self.assertFalse(status.enabled)
        self.assertEqual(status.probability, 0.30)
        sub = self.service.subscribe(SUBJECT, URL)
        self.assertEqual(self.service.subscribe(SUBJECT, URL), sub)
        self.assertEqual(self.service.subscriptions(), [sub])
        self.assertTrue(self.service.poll().disabled)
        sink = Mock()
        self.assertEqual(self.service.deliver(sink).skipped, "disabled")
        self.assertIsNone(self.service.claim())
        sink.assert_not_called()
        self.fetcher.assert_not_called()
        self.assertEqual(self.new_service().status(), self.service.status())

    def test_no_injected_fetcher_means_no_network_even_enabled(self):
        service = self.new_service(fetcher=None)
        service.configure(enabled=True)
        sub = service.subscribe(SUBJECT, URL)
        result = service.poll()
        self.assertEqual(result.polled, 0)
        self.assertEqual(result.errors, {sub.id: "no feed fetcher configured"})
        self.assertEqual(service.status().unread, 0)

    def test_enabled_without_subscriptions_does_not_fetch_or_draw(self):
        rng = Mock(side_effect=AssertionError("unexpected draw"))
        service = self.new_service(rng=rng)
        service.configure(enabled=True)
        self.assertEqual(service.poll().polled, 0)
        self.assertEqual(service.deliver(Mock()).skipped, "empty")
        self.fetcher.assert_not_called()
        rng.assert_not_called()

    def test_e2e_duplicate_restart_disable_probabilities_and_ack(self):
        sub = self.ready(probability=0.30)
        self.assertEqual(self.service.poll().duplicates, 1)
        self.service = self.new_service(rng=lambda: 0.3)
        self.assertEqual(self.service.poll().enqueued, 0)
        self.assertEqual(self.service.status().unread, 1)
        self.assertEqual(self.service.deliver(Mock()).skipped, "probability")
        self.service.configure(enabled=False)
        before = self.fetcher.call_count
        self.assertTrue(self.service.poll().disabled)
        self.assertEqual(self.service.deliver(Mock()).skipped, "disabled")
        self.assertEqual(self.fetcher.call_count, before)
        self.service.configure(enabled=True, probability=0)
        self.assertEqual(self.service.deliver(Mock()).skipped, "probability")
        self.service.configure(probability=0.30)
        self.service.rng = lambda: 0.299999
        received = []
        result = self.service.deliver(received.append)
        self.assertEqual(result.delivered, received)
        topic = received[0]
        self.assertEqual(topic.subscription_id, sub.id)
        self.assertEqual(topic.subject_id, SUBJECT)
        self.assertEqual(topic.source_url, "https://example.test/episodes/1")
        self.assertTrue(topic.created_at)
        self.assertTrue(topic.item_key)
        self.assertNotIn("Episode 1", topic.content)  # no plot/title injection
        self.assertNotIn("instruction", topic.content)
        self.service = self.new_service()
        self.assertEqual(self.service.status().read, 1)
        self.assertEqual(self.service.status().unread, 0)
        self.assertEqual(self.service.poll().duplicates, 1)
        self.assertEqual(self.service.deliver(Mock()).skipped, "empty")
        self.fetcher.return_value = ATOM
        self.assertEqual(self.service.poll().enqueued, 1)
        self.service.configure(probability=1)
        self.service.rng = Mock(side_effect=AssertionError("100% needs no RNG"))
        self.assertEqual(len(self.service.deliver(received.append).delivered), 1)
        self.assertEqual(self.service.status().read, 2)
        self.assertEqual(received[-1].source_url, "https://feeds.example.test/episodes/2")
        self.fetcher.assert_called_with(URL, timeout=FETCH_TIMEOUT, max_bytes=MAX_FEED_BYTES)

    def test_configuration_persisted_and_partial_updates_preserve_fields(self):
        self.service.configure(enabled=True, probability=0.7)
        service = self.new_service()
        self.assertTrue(service.status().enabled)
        self.assertEqual(service.configure(enabled=False).probability, 0.7)
        self.assertFalse(service.configure(probability=0.8).enabled)
        self.assertEqual(asdict(service.status())["probability"], 0.8)
        for probability in [-1, 1.1, float("nan"), float("inf"), "0.3", True]:
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                service.configure(probability=probability)
        with self.assertRaises(ValueError):
            service.configure(enabled="false")

    def test_unsubscribe_cancels_pending_and_retains_identity_tombstones(self):
        sub = self.ready()
        claim = self.service.claim()
        self.assertTrue(self.service.unsubscribe(sub.id))
        self.assertFalse(self.service.unsubscribe(sub.id))
        self.assertFalse(self.service.unsubscribe("missing"))
        self.assertEqual(self.service.subscriptions(), [])
        self.assertFalse(self.service.ack(claim.id, claim.lease_token))
        self.assertEqual(self.service.status().cancelled, 1)
        before = self.fetcher.call_count
        self.assertEqual(self.service.poll().polled, 0)
        self.assertEqual(self.fetcher.call_count, before)
        self.assertEqual(self.service.deliver(Mock()).skipped, "empty")
        self.assertEqual(self.service.subscribe(SUBJECT, URL).id, sub.id)
        self.assertEqual(self.service.poll().duplicates, 1)
        self.assertIsNone(self.service.claim())
        self.fetcher.return_value = ATOM
        self.assertEqual(self.service.poll().enqueued, 1)

    def test_invalid_subscriptions_rejected_without_fetch(self):
        urls = ["file:///tmp/rss", "ftp://example.test/feed", "magnet:?xt=1",
                "http:///no-host", "https://user:pass@example.test/feed",
                "https://example.test/feed#fragment", "https://example.test:99999/",
                "https://example.test:0/", "https://example.test/\nfeed", "", "//host/path"]
        for url in urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.service.subscribe(SUBJECT, url)
        for subject in ["", "   ", "bad\x1b[31m", "a" * 513, None]:
            with self.subTest(subject=subject), self.assertRaises(ValueError):
                self.service.subscribe(subject, URL)
        self.fetcher.assert_not_called()

    def test_identity_prefers_guid_over_changed_title_link_and_timestamp(self):
        self.ready()
        self.fetcher.return_value = RSS.replace(b"Episode 1", b"Corrected title").replace(
            b"episodes/1", b"episodes/changed")
        self.assertEqual(self.service.poll().duplicates, 1)
        self.fetcher.return_value = ATOM
        self.assertEqual(self.service.poll().enqueued, 1)
        self.fetcher.return_value = ATOM.replace(b"Episode 2", b"Updated").replace(
            b"2026-10-14", b"2026-10-15")
        self.assertEqual(self.service.poll().duplicates, 1)

    def test_link_title_fallbacks_rss1_and_duplicate_items(self):
        self.service.subscribe(SUBJECT, URL)
        self.service.configure(enabled=True)
        self.fetcher.return_value = b"""<rss><channel>
        <item><title>A</title><link>/a</link></item>
        <item><title>A correction</title><link>/a</link></item>
        <item><title>B</title><pubDate>Monday</pubDate></item>
        <item><title>B</title><pubDate>Tuesday</pubDate></item>
        <item><enclosure url="https://example.test/media"/></item>
        </channel></rss>"""
        result = self.service.poll()
        self.assertEqual((result.enqueued, result.duplicates), (2, 2))
        self.fetcher.return_value = b"""<rdf:RDF
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns="http://purl.org/rss/1.0/">
        <item rdf:about="urn:test:3"><title>C</title><link>/c</link></item></rdf:RDF>"""
        self.assertEqual(self.service.poll().enqueued, 1)
        self.assertEqual(self.service.poll().duplicates, 1)

    def test_item_ids_are_scoped_to_subscriptions_only_explicit_urls_polled(self):
        self.ready()
        second = "https://feeds.example.test/other.xml"
        self.service.subscribe("bgm:subject:456", second)
        outcome = self.service.poll()
        self.assertEqual((outcome.enqueued, outcome.duplicates), (1, 1))
        self.assertEqual({call.args[0] for call in self.fetcher.call_args_list}, {URL, second})
        self.assertEqual(self.service.status().unread, 2)

    def test_untrusted_xml_is_bounded_and_no_partial_enqueue(self):
        self.service.configure(enabled=True)
        sub = self.service.subscribe(SUBJECT, URL)
        dtd = '<!DOCTYPE rss [<!ENTITY x "payload">]><rss><channel>&x;</channel></rss>'
        fixtures = [dtd.encode(), dtd.encode("utf-16"),
                    b'<!DOCTYPE rss SYSTEM "file:///secret"><rss/>',
                    b'<rss><channel><item><guid>x</guid></item></channel>',
                    b"<html><body>not a feed</body></html>",
                    b"<rss>" + b"<x>" * 33 + b"</x>" * 33 + b"</rss>",
                    b"<rss>" + b"<x/>" * 20000 + b"</rss>",
                    b"<rss><channel>" + b"<item><guid>x</guid></item>" * 501
                    + b"</channel></rss>", b"x" * (MAX_FEED_BYTES + 1), "not bytes"]
        for payload in fixtures:
            with self.subTest(size=len(payload)):
                self.fetcher.return_value = payload
                result = self.service.poll()
                self.assertIn(sub.id, result.errors)
                self.assertEqual(self.service.status().unread, 0)

    def test_no_enclosure_or_unsafe_link_fetch_and_control_text_is_cleaned(self):
        self.service.subscribe(SUBJECT, URL)
        self.service.configure(enabled=True, probability=1)
        self.fetcher.return_value = b"""<rss><channel><item><guid>x</guid>
        <title>A&#x202E;B</title><link>javascript:alert(1)</link>
        <enclosure url="file:///private"/></item></channel></rss>"""
        self.assertEqual(self.service.poll().enqueued, 1)
        topic = self.service.claim()
        self.assertEqual(topic.source_url, URL)
        self.assertEqual(topic.title, "A B")
        self.assertEqual(self.fetcher.call_count, 1)

    def test_feed_failure_isolated_and_sensitive_exception_not_leaked(self):
        self.service.configure(enabled=True)
        bad = self.service.subscribe(SUBJECT, URL)
        good = self.service.subscribe("bgm:subject:456", "https://feeds.example.test/good")
        self.fetcher.side_effect = [TimeoutError("secret-token-in-url"), RSS]
        result = self.service.poll()
        self.assertEqual(result.errors, {bad.id: "TimeoutError"})
        self.assertEqual(result.enqueued, 1)
        self.assertNotIn("secret-token", str(asdict(result)))
        self.assertNotIn(good.id, result.errors)

    def test_elapsed_fetch_budget_rejects_even_successful_late_response(self):
        self.service.configure(enabled=True)
        sub = self.service.subscribe(SUBJECT, URL)
        with patch("mikirin.proactive.time.monotonic", side_effect=[0, FETCH_TIMEOUT + 1]):
            result = self.service.poll()
        self.assertEqual(result.errors, {sub.id: "TimeoutError"})
        self.assertEqual(self.service.status().unread, 0)

    def test_disable_or_unsubscribe_during_fetch_discards_result(self):
        self.service.configure(enabled=True)
        sub = self.service.subscribe(SUBJECT, URL)

        def disabling(url, **kwargs):
            self.new_service().configure(enabled=False)
            return RSS

        self.fetcher.side_effect = disabling
        self.assertTrue(self.service.poll().disabled)
        self.assertEqual(self.service.status().unread, 0)
        self.service.configure(enabled=True)

        def removing(url, **kwargs):
            self.new_service().unsubscribe(sub.id)
            return RSS

        self.fetcher.side_effect = removing
        self.assertEqual(self.service.poll().enqueued, 0)
        self.assertEqual(self.service.status().unread, 0)

    def test_failed_delivery_false_exception_and_interrupt_remain_retryable(self):
        self.ready()
        self.assertEqual(len(self.service.deliver(lambda _: False).failed), 1)
        self.assertEqual(self.service.status().leased, 0)
        failing = Mock(side_effect=RuntimeError("oops"))
        self.assertEqual(len(self.service.deliver(failing).failed), 1)
        with self.assertRaises(KeyboardInterrupt):
            self.service.deliver(Mock(side_effect=KeyboardInterrupt))
        self.assertEqual(self.service.status().unread, 1)
        self.assertEqual(self.service.status().read, 0)
        self.service = self.new_service()
        result = self.service.deliver(lambda _: True)
        self.assertEqual(result.delivered[0].attempts, 4)
        self.assertEqual(self.service.status().read, 1)

    def test_lease_recovery_fences_stale_ack_and_release(self):
        self.ready()
        claim = self.service.claim(lease_seconds=10)
        self.assertEqual(self.service.status().leased, 1)
        self.assertIsNone(self.new_service().claim())
        self.assertFalse(self.service.ack(claim.id, "wrong"))
        self.clock.now += 10
        self.assertFalse(self.service.ack(claim.id, claim.lease_token))
        retry = self.new_service().claim()
        self.assertEqual(retry.id, claim.id)
        self.assertNotEqual(retry.lease_token, claim.lease_token)
        self.assertEqual(retry.attempts, 2)
        self.assertFalse(self.service.release(claim.id, claim.lease_token))
        self.assertFalse(self.service.ack(claim.id, claim.lease_token))
        self.assertTrue(self.service.ack(retry.id, retry.lease_token))
        self.assertFalse(self.service.ack(retry.id, retry.lease_token))
        self.assertEqual(self.service.status().unread, 0)

    def test_crash_boundary_display_then_expiry_can_repeat_display(self):
        self.ready()
        sink = []
        claim = self.service.claim()
        sink.append(claim.id)  # simulate display followed by process death, without ack
        self.clock.now += 61
        self.new_service().deliver(lambda topic: sink.append(topic.id))
        self.assertEqual(sink, [claim.id, claim.id])  # explicitly NOT exactly-once
        self.assertEqual(self.service.status().read, 1)

    def test_callback_lease_expiry_is_reported_without_false_ack(self):
        self.ready()

        def slow_display(topic):
            self.clock.now += 61

        result = self.service.deliver(slow_display)
        self.assertFalse(result.delivered)
        self.assertIn("display may have occurred", next(iter(result.failed.values())))
        self.assertEqual(self.service.status().unread, 1)

    def test_callback_can_disable_without_lock_and_ack_already_displayed_item(self):
        self.ready()
        self.fetcher.return_value = ATOM
        self.service.poll()

        def display(topic):
            self.new_service().configure(enabled=False)

        result = self.service.deliver(display, limit=2)
        self.assertEqual(len(result.delivered), 1)
        self.assertEqual(result.skipped, "disabled")
        self.assertEqual(self.service.status().unread, 1)

    def test_one_probability_draw_per_batch_and_no_immediate_failed_retry(self):
        self.ready(probability=0.30)
        self.fetcher.return_value = ATOM
        self.service.poll()
        rng = Mock(return_value=0.1)
        self.service.rng = rng
        result = self.service.deliver(lambda _: True, limit=2)
        self.assertEqual(len(result.delivered), 2)
        rng.assert_called_once_with()
        self.fetcher.return_value = RSS.replace(b"episode-1", b"episode-3")
        self.service.poll()
        sink = Mock(return_value=False)
        self.assertEqual(len(self.service.deliver(sink, limit=100).failed), 1)
        sink.assert_called_once()

    def test_concurrent_poll_and_claim_use_sqlite_uniqueness_and_write_lock(self):
        self.service.subscribe(SUBJECT, URL)
        self.service.configure(enabled=True, probability=1)
        barrier = threading.Barrier(2)

        def fetching(url, **kwargs):
            barrier.wait(timeout=5)
            return RSS

        services = [self.new_service(fetcher=fetching), self.new_service(fetcher=fetching)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda s: s.poll(), services))
        self.assertEqual(sum(r.enqueued for r in results), 1)
        self.assertFalse(any(r.errors for r in results))
        barrier = threading.Barrier(2)

        def claiming(service):
            barrier.wait(timeout=5)
            return service.claim()

        with ThreadPoolExecutor(max_workers=2) as pool:
            claims = list(pool.map(claiming, services))
        self.assertEqual(sum(c is not None for c in claims), 1)

    def test_existing_user_store_progress_and_fts_untouched(self):
        store = UserStore(self.path)
        store.update_progress(subject_entity_id=SUBJECT, episode_entity_id=None,
                              progress_value=3, status="watching", source_message_id=None,
                              confidence=1)
        with closing(sqlite3.connect(self.path)) as db:
            before = db.execute("SELECT * FROM memory_note").fetchall()
            before_fts = db.execute("SELECT * FROM memory_note_fts").fetchall()
        self.service = self.new_service()
        self.service.configure(enabled=True)
        self.assertEqual(self.service.poll().polled, 0)  # watch_progress is not consent
        self.ready()
        self.service.deliver(lambda _: True)
        with closing(sqlite3.connect(self.path)) as db:
            self.assertEqual(db.execute("SELECT * FROM memory_note").fetchall(), before)
            self.assertEqual(db.execute("SELECT * FROM memory_note_fts").fetchall(), before_fts)
        self.assertEqual(store.progress(SUBJECT).progress_value, 3)

    def test_claim_and_delivery_validation(self):
        for lease in [0, -1, 3601, float("nan"), float("inf")]:
            with self.subTest(lease=lease), self.assertRaises(ValueError):
                self.service.claim(lease_seconds=lease)
        for limit in [0, -1, 101, 1.5, True]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.service.deliver(Mock(), limit=limit)
        self.ready(probability=0.3)
        for draw in [-1, 1, float("nan")]:
            self.service.rng = lambda: draw
            with self.subTest(draw=draw), self.assertRaises(ValueError):
                self.service.claim()
        self.assertEqual(self.service.status().leased, 0)

    def test_feed_enqueue_rollback_does_not_report_uncommitted_messages(self):
        self.service.subscribe(SUBJECT, URL)
        self.service.configure(enabled=True, probability=1)
        self.fetcher.return_value = b"""<rss><channel>
        <item><guid>one</guid><title>First</title></item>
        <item><guid>two</guid><title>Second</title></item></channel></rss>"""
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""CREATE TRIGGER fail_second BEFORE INSERT ON proactive_topic
                WHEN NEW.title='Second' BEGIN SELECT RAISE(ABORT, 'simulated failure'); END""")
        result = self.service.poll()
        self.assertEqual(result.enqueued, 0)
        self.assertEqual(result.duplicates, 0)
        self.assertTrue(result.errors)
        self.assertEqual(self.service.status().unread, 0)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DROP TRIGGER fail_second")
        self.assertEqual(self.new_service().poll().enqueued, 2)

    def test_failed_ack_leaves_atomic_unread_state_and_recoverable_lease(self):
        self.ready()
        claim = self.service.claim()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""CREATE TRIGGER fail_ack BEFORE UPDATE OF read_at ON proactive_topic
                BEGIN SELECT RAISE(ABORT, 'simulated failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.service.ack(claim.id, claim.lease_token)
        self.assertEqual(self.service.status().read, 0)
        self.assertEqual(self.service.status().unread, 1)
        self.assertEqual(self.service.status().leased, 1)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DROP TRIGGER fail_ack")
        self.assertTrue(self.new_service().ack(claim.id, claim.lease_token))

    def test_invalid_callback_result_cannot_acknowledge_work(self):
        self.ready()
        outcome = self.service.deliver(lambda _: "not an acknowledgment")
        self.assertEqual(next(iter(outcome.failed.values())), "TypeError")
        self.assertEqual(self.service.status().unread, 1)
        self.assertEqual(self.service.status().leased, 0)


class RSSIntegrationReviewTests(unittest.TestCase):
    """Read-only integration review; expected failures document unfixed UI bugs.

    Remove expectedFailure when the corresponding CLI/TUI fix lands. All input,
    feeds and databases are synthetic; socket/DNS guards forbid real traffic.
    """

    def setUp(self):
        from mikirin.cli import build_parser
        from mikirin.layout import ProjectLayout

        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.layout = ProjectLayout(self.root)
        self.database = self.root / "explicit-content.db"
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("CREATE TABLE entity(id TEXT PRIMARY KEY, entity_type TEXT)")
            db.execute("INSERT INTO entity VALUES(?, 'subject')", (SUBJECT,))
        self.parser = build_parser()
        self.service = ProactiveService(self.layout.user_db, fetcher=lambda *a, **kw: RSS)
        self.output = self.enterContext(redirect_stdout(io.StringIO()))
        for target in ("socket.socket", "socket.getaddrinfo"):
            guard = self.enterContext(patch(target, side_effect=AssertionError("live network")))
            self.addCleanup(guard.assert_not_called)

    def command(self, *arguments):
        args = self.parser.parse_args(["--root", str(self.root), "rss", *arguments])
        return args.handler(args)

    def seed(self):
        self.service.subscribe(SUBJECT, URL)
        self.service.configure(enabled=True, probability=1)
        self.assertEqual(self.service.poll().enqueued, 1)

    def console(self, *, browse_only=False):
        from mikirin.tui import AgentConsole

        return AgentConsole(self.layout, self.database, browse_only=browse_only)

    def test_cli_explicit_database_subscribe_and_disabled_poll_never_fetch(self):
        with patch("mikirin.proactive.fetch_http_feed") as fetch:
            self.assertEqual(self.command("subscribe", SUBJECT, URL,
                                          "--database", str(self.database)), 0)
            self.assertEqual(self.command("poll"), 0)
        fetch.assert_not_called()
        self.assertEqual(self.service.subscriptions()[0].subject_id, SUBJECT)
        self.assertFalse(self.service.status().enabled)

    def test_tui_startup_delivers_once_without_poll_and_browse_blocks_poll(self):
        self.seed()
        console = self.console(browse_only=True)
        original_deliver = ProactiveService.deliver
        with patch.object(ProactiveService, "poll") as poll, patch.object(
            ProactiveService, "deliver", autospec=True, side_effect=original_deliver,
        ) as deliver, patch("builtins.input", side_effect=["/rss status", "/rss poll", EOFError]):
            self.assertEqual(console.run(), 0)
        poll.assert_not_called()
        self.assertEqual(deliver.call_count, 1)
        self.assertEqual(self.service.status().read, 1)
        self.assertIn("不访问 RSS 网络", self.output.getvalue())

    def test_review_tui_subscribe_should_inherit_current_explicit_database(self):
        console = self.console()
        # The active content DB exists, but there is no published current.json.
        # Current code wrongly calls database_for() without console.database.
        console.handle_command(f"/rss subscribe {SUBJECT} {URL}")
        self.assertEqual(self.service.subscriptions()[0].subject_id, SUBJECT)

    def test_review_startup_topic_should_be_in_next_turn_replay(self):
        self.seed()
        console = self.console()
        with patch("builtins.input", side_effect=EOFError):
            self.assertEqual(console.run(), 0)
        self.assertEqual(self.service.status().read, 1)
        self.assertIn("有新更新", self.output.getvalue())
        # chat() builds the next model history from this exact replay API.
        replay = console.session_store.replay_messages(console.session_id)
        self.assertTrue(any(getattr(message, "role", None) == "assistant" for message in replay))

    def test_review_failed_scheduled_poll_should_exit_nonzero(self):
        self.seed()
        with patch("mikirin.proactive.fetch_http_feed", side_effect=TimeoutError):
            code = self.command("poll")
        self.assertIn("TimeoutError", self.output.getvalue())
        self.assertNotEqual(code, 0)

    def test_review_cli_deliver_should_emit_one_json_result_not_double_content(self):
        self.seed()
        self.assertEqual(self.command("deliver"), 0)
        self.assertEqual(self.service.status().read, 1)
        # Other RSS subcommands emit one JSON result. Currently a plain-text
        # topic is followed by JSON containing the same content a second time.
        payload = json.loads(self.output.getvalue())
        self.assertEqual(len(payload["delivered"]), 1)


class HTTPFeedReaderTests(unittest.TestCase):
    def response(self, *, data=RSS, headers=None, status=200, url=URL):
        response = Mock()
        response.status = status
        response.headers = headers or {}
        response.geturl.return_value = url
        response.read1.side_effect = [data, b""]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        return response

    def test_bounded_feed_read_only_and_closed(self):
        response = self.response()
        with patch("mikirin.proactive.build_opener") as build:
            build.return_value.open.return_value = response
            self.assertEqual(fetch_http_feed(URL, timeout=5, max_bytes=MAX_FEED_BYTES), RSS)
        request = build.return_value.open.call_args.args[0]
        self.assertEqual(request.full_url, URL)
        self.assertEqual(build.return_value.open.call_args.kwargs, {"timeout": 5})
        self.assertEqual(build.call_args.args[0].proxies, {})
        self.assertIsInstance(build.call_args.args[1], _NoRedirect)
        self.assertTrue(all(call.args[0] <= 65536 for call in response.read1.call_args_list))
        response.__exit__.assert_called_once()

    def test_redirect_handler_refuses_even_http_redirects(self):
        with self.assertRaises(ValueError):
            _NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.test/")

    def test_unsafe_responses_size_and_timeout_rejected(self):
        cases = [self.response(headers={"Content-Length": str(MAX_FEED_BYTES + 1)}),
                 self.response(headers={"Content-Encoding": "gzip"}),
                 self.response(status=302), self.response(url="https://other.test/"),
                 self.response(data=b"x" * (MAX_FEED_BYTES + 1))]
        for response in cases:
            with self.subTest(response=response), patch("mikirin.proactive.build_opener") as build:
                build.return_value.open.return_value = response
                with self.assertRaises(ValueError):
                    fetch_http_feed(URL, timeout=5, max_bytes=MAX_FEED_BYTES)
                response.__exit__.assert_called_once()
        response = self.response()
        with patch("mikirin.proactive.build_opener") as build:
            build.return_value.open.return_value = response
            with patch("mikirin.proactive.time.monotonic", side_effect=[0, 6]):
                with self.assertRaises(TimeoutError):
                    fetch_http_feed(URL, timeout=5, max_bytes=MAX_FEED_BYTES)

    def test_invalid_url_never_constructs_network_reader(self):
        with patch("mikirin.proactive.build_opener") as build:
            with self.assertRaises(ValueError):
                fetch_http_feed("file:///private", timeout=5, max_bytes=100)
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()

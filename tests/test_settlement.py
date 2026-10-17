import json
import sqlite3
from contextlib import closing
import tempfile
import unittest
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from pathlib import Path

from mikirin.settlement import MemorySettlement, SettlementBusyError, SettlementValidationError
from mikirin.user_store import UserStore
from mikirin.layout import ProjectLayout
from mikirin.prompts import PromptBuilder


class FakeClient:
    def __init__(self, response: dict) -> None:
        self.payload = response
        self.calls: list[list[dict]] = []

    def complete_json(self, messages, **kwargs):
        self.calls.append(messages)
        return {"_fake": True}

    def response_content(self, payload):
        return json.dumps(self.payload, ensure_ascii=False)


def make_user_db(tmp: Path) -> Path:
    path = tmp / "user.db"
    UserStore(path)
    return path


class SettlementTests(unittest.TestCase):
    def test_add_and_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            user_db = make_user_db(Path(d))
            flash = FakeClient({"facts": [
                {"content": "用户孤独摇滚看到第8集", "keywords": "孤独摇滚 进度",
                 "entity_hint": "孤独摇滚"},
            ]})
            pro = FakeClient({"action": "add", "target_note_id": None,
                              "reason": "无冲突"})
            settlement = MemorySettlement(user_db, flash, pro)
            messages = [{"role": "user", "content": "孤独摇滚我看到第8集了"}]
            report = settlement.settle("s1", messages)
            self.assertEqual(report.notes_added, 1)
            # 重复结算幂等
            again = settlement.settle("s1", messages)
            self.assertTrue(again.already_settled)
            with closing(sqlite3.connect(user_db)) as db:
                count = db.execute("SELECT COUNT(*) FROM memory_note").fetchone()[0]
            self.assertEqual(count, 1)

    def test_update_expires_old_note(self):
        with tempfile.TemporaryDirectory() as d:
            user_db = make_user_db(Path(d))
            with closing(sqlite3.connect(user_db)) as db:
                db.execute(
                    "INSERT INTO memory_note(id, content, keywords, context,"
                    " entity_ids_json, valid_from, source_session_id, created_at)"
                    " VALUES('note:old', '孤独摇滚看到第5集', '孤独摇滚', '', '[]',"
                    " '2026-01-01', 's0', '2026-01-01')")
                db.commit()
            flash = FakeClient({"facts": [
                {"content": "用户孤独摇滚看到第8集", "keywords": "孤独摇滚",
                 "entity_hint": "孤独摇滚"}]})
            pro = FakeClient({"action": "update", "target_note_id": "note:old",
                              "reason": "进度推进"})
            settlement = MemorySettlement(user_db, flash, pro)
            report = settlement.settle("s1", [{"role": "user", "content": "看到第8集"}])
            self.assertEqual(report.notes_expired, 1)
            with closing(sqlite3.connect(user_db)) as db:
                row = db.execute(
                    "SELECT valid_until FROM memory_note WHERE id='note:old'"
                ).fetchone()
                self.assertIsNotNone(row[0])

    def test_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            user_db = make_user_db(Path(d))
            flash = FakeClient({"facts": [
                {"content": "用户讨厌胃药番", "keywords": "雷点", "entity_hint": ""}]})
            pro = FakeClient({"action": "add", "target_note_id": None, "reason": ""})
            settlement = MemorySettlement(user_db, flash, pro)
            settlement.settle("s1", [{"role": "user", "content": "x"}], dry_run=True)
            with closing(sqlite3.connect(user_db)) as db:
                count = db.execute("SELECT COUNT(*) FROM memory_note").fetchone()[0]
            self.assertEqual(count, 0)

    def test_skip_when_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            user_db = make_user_db(Path(d))
            with closing(sqlite3.connect(user_db)) as db:
                db.execute(
                    "INSERT INTO memory_note(id, content, keywords, context,"
                    " entity_ids_json, valid_from, source_session_id, created_at)"
                    " VALUES('note:old', '用户是京阿尼铁粉', '京阿尼', '', '[]',"
                    " '2026-01-01', 's0', '2026-01-01')")
                db.commit()
            flash = FakeClient({"facts": [
                {"content": "用户是京阿尼铁粉", "keywords": "京阿尼", "entity_hint": ""}]})
            pro = FakeClient({"action": "skip", "target_note_id": None,
                              "reason": "语义重复"})
            settlement = MemorySettlement(user_db, flash, pro)
            report = settlement.settle("s1", [{"role": "user", "content": "我爱京阿尼"}])
            self.assertEqual(report.notes_added, 0)
            self.assertEqual(report.skipped, 1)


class DurableSettlementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = make_user_db(self.root)
        self.messages = [{"role": "user", "content": "我喜欢温柔的故事"}]

    def worker(self, facts=None, action="add", target=None, **kwargs):
        flash = FakeClient({"facts": facts if facts is not None else [
            {"content": "用户喜欢温柔的故事", "keywords": "喜好", "entity_hint": ""},
        ]})
        pro = FakeClient({"action": action, "target_note_id": target, "reason": "test"})
        return MemorySettlement(self.db, flash, pro, **kwargs)

    def seed(self, note_id="old", entity="anime:A", content="旧的偏好", expired=None):
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("INSERT INTO memory_note(id, content, entity_ids_json, valid_from, "
                       "valid_until, created_at) VALUES(?,?,?,'2020-01-01',?,'2020-01-01')",
                       (note_id, content, json.dumps([entity] if entity else []), expired))

    def count(self, table):
        with closing(sqlite3.connect(self.db)) as db:
            return db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    def test_queue_without_clients_and_duplicate_enqueue(self):
        queue = MemorySettlement(self.db)
        key = queue.enqueue("s", self.messages)
        self.assertEqual(queue.enqueue("s", self.messages), key)
        self.assertEqual(queue.queue_status()[0]["status"], "pending")
        self.assertEqual(len(self.worker().run_pending()), 1)
        self.assertEqual(queue.queue_status()[0]["status"], "done")

    def test_full_session_bounded_chunks_including_long_turn_tail(self):
        worker = self.worker(chunk_chars=128)
        transcript = [{"role": "user", "content": "首" + "长" * 24000 + "最后偏好"},
                      {"role": "assistant", "content": "最后回答"}]
        worker.settle("s", transcript)
        seen = [call[1]["content"] for call in worker.flash.calls]
        self.assertTrue(all(len(chunk) <= 128 for chunk in seen))
        self.assertIn("最后偏好", "".join(seen))
        self.assertIn("assistant: 最后回答", "".join(seen))
        self.assertEqual(sum(chunk.count("长") for chunk in seen), 24000)
        self.assertEqual(self.count("memory_note"), 1)
        self.assertEqual(len(worker.pro.calls), 1)

    def test_empty_result_is_durably_done(self):
        worker = self.worker(facts=[])
        worker.settle("s", self.messages)
        self.assertTrue(worker.settle("s", self.messages).already_settled)
        self.assertEqual(len(worker.flash.calls), 1)
        self.assertEqual(self.count("memory_note"), 0)

    def test_content_receipt_survives_changed_transcript_and_keywords(self):
        worker = self.worker()
        worker.settle("s", self.messages)
        worker.flash.payload["facts"][0]["keywords"] = "不同的关键词"
        repeated = worker.settle("s", self.messages + [{"role": "assistant", "content": "明白"}])
        self.assertEqual(repeated.notes_added, 0)
        self.assertEqual(repeated.skipped, 1)
        self.assertEqual(self.count("memory_note"), 1)
        self.assertEqual(self.count("memory_settlement_receipt"), 1)
        self.assertEqual(len(worker.pro.calls), 1)

    def test_all_four_actions_and_active_only_next_session_prefetch(self):
        for action in ("add", "update", "invalidate", "coexist"):
            with self.subTest(action=action):
                entity = f"anime:{action}"
                target = "old-" + action
                self.seed(target, entity, "旧偏好-" + action)
                worker = self.worker([{"content": "新偏好-" + action,
                                       "entity_hint": entity}], action,
                                     target if action in ("update", "invalidate") else None)
                report = worker.settle(action, self.messages)
                self.assertEqual(report.notes_added, 1)
                self.assertEqual(report.notes_expired, int(action in ("update", "invalidate")))
                active = UserStore(self.db).active_memory_notes(entity_ids=[entity])
                self.assertEqual(len(active), 1 if action in ("update", "invalidate") else 2)
        layout = ProjectLayout(self.root)
        layout.initialize()
        # Put the existing temporary DB at the layout's own user database path.
        with closing(sqlite3.connect(self.db)) as source:
            with closing(sqlite3.connect(layout.user_db)) as destination:
                source.backup(destination)
        prompt = PromptBuilder(layout).build().volatile
        self.assertIn("新偏好-update", prompt)
        self.assertNotIn("旧偏好-update", prompt)
        self.assertNotIn("旧偏好-invalidate", prompt)
        self.assertIn("旧偏好-coexist", prompt)

    def test_forged_cross_entity_expired_and_missing_targets_fail_closed(self):
        self.seed("other", "anime:B")
        self.seed("expired", "anime:A", expired="2021-01-01")
        for target in ("invented", "other", "expired", None):
            with self.subTest(target=target):
                worker = self.worker([{"content": "新事实", "entity_hint": "anime:A"}],
                                     "update", target)
                with self.assertRaises(SettlementValidationError):
                    worker.settle(str(target), self.messages)
        self.assertEqual(self.count("memory_note"), 2)
        self.assertEqual(self.count("memory_settlement_receipt"), 0)

    def test_expiration_and_insert_rollback_together(self):
        self.seed()
        worker = self.worker([{"content": "新事实", "entity_hint": "anime:A"}], "update", "old")
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("CREATE TRIGGER fail_new BEFORE INSERT ON memory_note "
                       "BEGIN SELECT RAISE(ABORT, 'simulated crash before insert'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            worker.settle("s", self.messages)
        self.assertEqual(len(UserStore(self.db).active_memory_notes()), 1)
        self.assertEqual(self.count("memory_settlement_receipt"), 0)
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("DROP TRIGGER fail_new")
        self.assertEqual(worker.run_pending()[0].notes_expired, 1)
        self.assertEqual(self.count("memory_link"), 1)
        self.assertEqual(len(worker.flash.calls), 1)

    def test_bad_json_is_failed_not_successful_empty_and_retry_works(self):
        worker = self.worker()
        worker.flash.payload = {"facts": "invalid"}
        with self.assertRaises(SettlementValidationError):
            worker.settle("s", self.messages)
        self.assertEqual(worker.queue_status()[0]["status"], "failed")
        good = self.worker()
        self.assertEqual(len(good.run_pending()), 1)
        self.assertEqual(good.queue_status()[0]["attempts"], 2)

    def test_extraction_checkpoint_survives_timeout(self):
        worker = self.worker(chunk_chars=128)
        original = worker.flash.complete_json
        def timeout_second(messages, **kwargs):
            if len(worker.flash.calls) == 1:
                raise TimeoutError("test only")
            return original(messages, **kwargs)
        worker.flash.complete_json = timeout_second
        with self.assertRaises(TimeoutError):
            worker.settle("s", [{"role": "user", "content": "x" * 180}])
        self.assertEqual(self.count("memory_settlement_chunk"), 1)
        restart = self.worker(chunk_chars=1000)
        restart.run_pending()
        self.assertEqual(len(restart.flash.calls), 1)
        self.assertEqual(self.count("memory_settlement_chunk"), 2)

    def test_process_exit_after_committed_fact_recovers_without_reextracting(self):
        facts = [{"content": "第一偏好"}, {"content": "第二偏好"}]
        worker = self.worker(facts)
        worker.enqueue("s", self.messages)
        script = '''
import json, os, sys
from pathlib import Path
from mikirin.settlement import MemorySettlement
class Fake:
    def __init__(self, value): self.value = value
    def complete_json(self, *args, **kwargs): return self.value
    def response_content(self, payload): return json.dumps(payload)
class Crash(MemorySettlement):
    def _apply_fact(self, *args, **kwargs):
        result = super()._apply_fact(*args, **kwargs)
        os._exit(17)
Crash(Path(sys.argv[1]), Fake({"facts": [{"content": "第一偏好"}, {"content": "第二偏好"}]}),
      Fake({"action": "add"})).run_pending()
'''
        result = subprocess.run([sys.executable, "-c", script, str(self.db)],
                                capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(self.count("memory_note"), 1)
        self.assertEqual(worker.queue_status()[0]["status"], "running")
        report = worker.run_pending()[0]
        self.assertEqual(report.notes_added, 2)
        self.assertEqual(len(worker.flash.calls), 0)
        self.assertEqual(len(worker.pro.calls), 1)
        self.assertEqual(self.count("memory_note"), 2)
        self.assertEqual(self.count("memory_settlement_receipt"), 2)

    def test_single_worker_lock_does_not_block_progress_or_lose_second_job(self):
        entered, release = Event(), Event()
        worker = self.worker()
        original = worker.flash.complete_json
        def blocked(messages, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test synchronization")
            return original(messages, **kwargs)
        worker.flash.complete_json = blocked
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(worker.settle, "s1", self.messages)
            try:
                self.assertTrue(entered.wait(5))
                with self.assertRaises(SettlementBusyError):
                    self.worker().settle("s2", self.messages)
                UserStore(self.db).update_progress(subject_entity_id="anime", episode_entity_id=None,
                    progress_value=7, status="watching", source_message_id="m", confidence=1)
            finally:
                release.set()
            first.result(timeout=5)
        self.assertEqual(len(self.worker().run_pending()), 1)
        self.assertEqual([q["status"] for q in worker.queue_status()], ["done", "done"])

    def test_concurrent_memory_change_retries_instead_of_expiring_stale_target(self):
        self.seed()
        worker = self.worker([{"content": "新事实", "entity_hint": "anime:A"}], "update", "old")
        original = worker.pro.complete_json
        def change_during_request(messages, **kwargs):
            with closing(sqlite3.connect(self.db)) as db, db:
                db.execute("UPDATE memory_note SET valid_until='2021-01-01' WHERE id='old'")
            return original(messages, **kwargs)
        worker.pro.complete_json = change_during_request
        with self.assertRaises(SettlementValidationError):
            worker.settle("s", self.messages)
        self.assertEqual(self.count("memory_note"), 1)

    def test_dry_run_is_database_byte_identical(self):
        worker = self.worker()
        before = self.db.read_bytes()
        worker.settle("s", self.messages, dry_run=True)
        self.assertEqual(self.db.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

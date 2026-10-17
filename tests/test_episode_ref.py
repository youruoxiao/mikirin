import sqlite3
from contextlib import closing
import unittest
from pathlib import Path

from mikirin.episode_ref import (
    EpisodeRef,
    EpisodeTable,
    _cn_number,
    parse_episode_mentions,
)


def make_db(tmp: Path) -> Path:
    db_path = tmp / "t.db"
    with closing(sqlite3.connect(db_path)) as db:
        db.execute(
            "CREATE TABLE episodes(entity_id TEXT, subject_entity_id TEXT, "
            "sort REAL, type INTEGER, name TEXT, name_cn TEXT)"
        )
        db.executemany(
            "INSERT INTO episodes VALUES(?,?,?,?,?,?)",
            [
                ("e1", "s1", 1.0, 0, "ep1", ""),
                ("e2", "s1", 2.0, 0, "ep2", ""),
                ("e3", "s1", 3.0, 0, "ep3", ""),
                ("esp", "s1", 3.5, 1, "sp", ""),       # SP 不参与数字映射
                ("e11", "s1", 11.0, 0, "ep11", ""),
                ("e115", "s1", 11.5, 1, "sp2", ""),
            ],
        )
        db.commit()
    return db_path


class CnNumberTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(_cn_number("八"), 8)
        self.assertEqual(_cn_number("十"), 10)
        self.assertEqual(_cn_number("十一"), 11)
        self.assertEqual(_cn_number("二十"), 20)
        self.assertEqual(_cn_number("二十三"), 23)
        self.assertEqual(_cn_number("8"), 8)
        self.assertIsNone(_cn_number(""))
        self.assertIsNone(_cn_number("孤独"))


class MentionParseTests(unittest.TestCase):
    def test_episode_forms(self):
        numbers = [m.display_number for m in parse_episode_mentions("第8集")]
        self.assertEqual(numbers, [8])
        numbers = [m.display_number for m in parse_episode_mentions("第八话真神")]
        self.assertEqual(numbers, [8])
        numbers = [m.display_number for m in parse_episode_mentions("EP8 和 ep.12 对比")]
        self.assertEqual(numbers, [8, 12])

    def test_range_is_not_single_episode(self):
        mentions = parse_episode_mentions("前三集铺垫怎么样")
        self.assertEqual(len(mentions), 1)
        self.assertEqual(mentions[0].kind, "range_end")
        self.assertEqual(mentions[0].display_number, 3)

    def test_no_false_positive_on_plain_text(self):
        self.assertEqual(parse_episode_mentions("孤独摇滚好看吗"), [])
        self.assertEqual(parse_episode_mentions("这个乐队有几个人"), [])


class EpisodeTableTests(unittest.TestCase):
    def test_resolve_main_only(self):
        with tempfile_db(self) as path:
            with closing(sqlite3.connect(path)) as db:
                table = EpisodeTable.load(db, "s1")
        self.assertEqual(table.resolve_number(3).episode_id, "e3")
        self.assertIsNone(table.resolve_number(4))
        # SP (type=1, sort 3.5) 不占用 display 3
        self.assertEqual(table.resolve_number(3).ep_type, 0)


import tempfile
from contextlib import closing


class tempfile_db:
    def __init__(self, case: unittest.TestCase) -> None:
        self.case = case

    def __enter__(self) -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        return make_db(Path(self.tmp.__enter__()))

    def __exit__(self, *args) -> None:
        self.tmp.__exit__(*args)


if __name__ == "__main__":
    unittest.main()

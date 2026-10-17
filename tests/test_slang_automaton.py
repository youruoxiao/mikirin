import unittest

from mikirin.slang.automaton import (
    AhoCorasick,
    AliasRecord,
    compile_entity_aliases,
    scan_entity_aliases,
)


class SlangAutomatonTests(unittest.TestCase):
    def test_one_scan_erases_known_aliases_and_returns_cooccurrence(self) -> None:
        aliases = [
            AliasRecord("bgm:character:1", "电锯人"),
            {"entity_id": "bgm:character:2", "alias_text": "团长"},
            ("bgm:subject:3", "Foo"),
        ]
        result = compile_entity_aliases(aliases).scan("我看电锯人、团长和FOO")
        self.assertEqual(
            result.entity_ids,
            ("bgm:character:1", "bgm:character:2", "bgm:subject:3"),
        )
        self.assertEqual(
            [(item.entity_id_a, item.entity_id_b) for item in result.cooccurrence],
            [
                ("bgm:character:1", "bgm:character:2"),
                ("bgm:character:1", "bgm:subject:3"),
                ("bgm:character:2", "bgm:subject:3"),
            ],
        )
        self.assertNotIn("电锯人", result.erased_text)
        self.assertNotIn("团长", result.erased_text)
        self.assertNotIn("FOO", result.erased_text.upper())
        self.assertEqual(len(result.erased_text), len(result.text))

    def test_overlapping_aliases_use_longest_leftmost_for_erasure(self) -> None:
        automaton = AhoCorasick(
            [
                ("e1", "电锯"),
                ("e1", "电锯人"),
                ("e2", "锯人"),
            ]
        )
        result = automaton.scan("电锯人")
        self.assertEqual(len(result.matches), 3)
        self.assertEqual([(m.start, m.end) for m in result.erased_matches], [(0, 3)])

    def test_unicode_normalization_and_empty_input(self) -> None:
        result = scan_entity_aliases("ＦＯＯとカナ", [("work", "foo"), ("work2", "カナ")])
        self.assertEqual(result.entity_ids, ("work", "work2"))
        empty = scan_entity_aliases("", [("work", "foo")])
        self.assertEqual(empty.matches, ())
        self.assertEqual(empty.cooccurrence, ())
        self.assertEqual(empty.erased_text, "")

    def test_duplicate_and_empty_alias_rows_are_ignored(self) -> None:
        automaton = AhoCorasick(
            [
                ("e1", "foo"),
                ("e1", "foo"),
                ("", "bar"),
                ("e2", ""),
            ]
        )
        self.assertEqual(len(automaton.aliases), 1)
        self.assertEqual(len(automaton.find_matches("foo foo")), 2)


if __name__ == "__main__":
    unittest.main()

import unittest

from mikirin.slang.candidates import (
    SLOT_PATTERNS,
    candidate_strings,
    find_slot_hits,
    generate_char_ngrams,
    generate_latin_kana_digit_runs,
    generate_slot_candidates,
)


class SlangCandidateTests(unittest.TestCase):
    def test_unicode_ngrams_do_not_cross_whitespace(self) -> None:
        spans = generate_char_ngrams("你好 世界", min_n=1, max_n=5)
        values = {span.text for span in spans}
        self.assertIn("你好", values)
        self.assertIn("世界", values)
        self.assertNotIn("好 世", values)
        self.assertEqual(generate_char_ngrams("", 1, 5), ())

    def test_latin_kana_and_unicode_digit_runs(self) -> None:
        spans = generate_latin_kana_digit_runs("ABC カナかな １２３ K-ON v2.0 中文")
        self.assertEqual(
            [span.text for span in spans],
            ["ABC", "カナかな", "１２３", "K-ON", "v2.0"],
        )

    def test_slot_inventory_is_explicit_and_covers_three_scripts(self) -> None:
        self.assertGreaterEqual(len(SLOT_PATTERNS), 30)
        text = "看电锯人、追团长、炮姐厨、foo season 2、見る葬送のフリーレン"
        hits = find_slot_hits(text)
        names = {hit.name for hit in hits}
        self.assertIn("cn_watch", names)
        self.assertIn("cn_follow", names)
        self.assertIn("cn_fan_suffix", names)
        self.assertIn("en_season", names)
        self.assertIn("jp_watch", names)
        self.assertIn("电锯人", {hit.candidate for hit in hits})
        self.assertTrue(generate_slot_candidates(text))

    def test_complete_candidate_keys_are_unique_and_deterministic(self) -> None:
        first = candidate_strings("K-ON 看电锯人")
        second = candidate_strings("K-ON 看电锯人")
        self.assertEqual(first, second)
        self.assertEqual(first, tuple(sorted(first, key=lambda value: (len(value), value))))

    def test_invalid_ngram_range_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_char_ngrams("abc", 0, 5)
        with self.assertRaises(ValueError):
            generate_char_ngrams("abc", 5, 1)


if __name__ == "__main__":
    unittest.main()

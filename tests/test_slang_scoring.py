import unittest
import math

from mikirin.slang.scoring import (
    BucketEvidence,
    CandidateObservation,
    DocumentRecord,
    LinkBucket,
    ScoreBand,
    build_corpus_stats,
    compute_features,
    link_bucket,
    observations_from_documents,
    rank_candidates,
    score_candidate,
)


class SlangScoringTests(unittest.TestCase):
    def test_min_pmi_uses_whole_word_all_splits_not_adjacent_pairs(self):
        # AB and BC are frequent separately; ABC is not equally frequent.
        stats = build_corpus_stats([
            CandidateObservation("甲乙丙", "doc", source_text="甲乙丙 甲乙 乙丙 甲乙 乙丙")
        ])
        actual = compute_features("甲乙丙", stats).min_pmi
        counts = stats.ngram_counts
        expected = min(math.log2(counts["甲乙丙"] * stats.total_characters /
                                (counts["甲乙丙"[:i]] * counts["甲乙丙"[i:]])) for i in (1, 2))
        self.assertAlmostEqual(actual, expected)
        adjacent = min(math.log2(counts[pair] * stats.total_characters /
                                (counts[pair[0]] * counts[pair[1]])) for pair in ("甲乙", "乙丙"))
        self.assertNotEqual(actual, adjacent)

    def _rows(self) -> list[CandidateObservation]:
        return [
            CandidateObservation(
                "电锯人",
                "d1",
                "alice",
                "看",
                "厨",
                "subject",
                ("cn_watch",),
                False,
                "看电锯人厨",
            ),
            CandidateObservation(
                "电锯人",
                "d2",
                "bob",
                "追",
                "很",
                "episode",
                ("cn_follow",),
                False,
                "追电锯人很爽",
            ),
            CandidateObservation(
                "电锯人",
                "d3",
                "alice",
                "看",
                "！",
                "subject",
                (),
                True,
                "看《电锯人》！",
            ),
            CandidateObservation("锯", "d1", "alice", "电", "人", "subject", (), False, "看电锯人厨"),
        ]

    def test_required_features_are_computed(self) -> None:
        stats = build_corpus_stats(self._rows())
        features = compute_features("电锯人", stats)
        self.assertEqual(features.occurrence_count, 3)
        self.assertEqual(features.document_frequency, 3)
        self.assertEqual(features.author_breadth, 2)
        self.assertEqual(features.slot_hits, 2)
        self.assertAlmostEqual(features.slot_hit_rate, 2 / 3)
        self.assertAlmostEqual(features.missing_title_bracket_rate, 2 / 3)
        self.assertGreater(features.left_adjacency_entropy, 0.0)
        self.assertGreater(features.context_distribution_entropy, 0.0)
        self.assertTrue(features.min_pmi > float("-inf"))

    def test_statistics_and_ranking_are_order_independent(self) -> None:
        rows = self._rows()
        forward = build_corpus_stats(rows)
        reverse = build_corpus_stats(reversed(rows))
        self.assertEqual(forward, reverse)
        ranked = rank_candidates(["电锯人", "锯", "电锯人"], forward)
        self.assertEqual([item.candidate for item in ranked], ["电锯人", "锯"])
        self.assertEqual(ranked, rank_candidates(["锯", "电锯人"], forward))

    def test_score_is_bounded_and_unknown_is_low(self) -> None:
        stats = build_corpus_stats(self._rows())
        known = score_candidate("电锯人", stats)
        unknown = score_candidate("不存在", stats)
        self.assertGreaterEqual(known.score, 0.0)
        self.assertLessEqual(known.score, 1.0)
        self.assertEqual(unknown.score, 0.0)
        self.assertIsInstance(known.bucket, ScoreBand)

    def test_document_adapter_handles_chinese_japanese_english_and_empty(self) -> None:
        observations = observations_from_documents(
            [
                DocumentRecord("cn", "看电锯人"),
                DocumentRecord("jp", "このアニメ第2期を見る"),
                DocumentRecord("en", "watch Frieren season 2"),
                DocumentRecord("empty", ""),
            ],
            min_n=1,
            max_n=2,
        )
        self.assertTrue(observations)
        self.assertTrue(any("cn_watch" in row.slot_names for row in observations))
        self.assertTrue(any("en_season" in row.slot_names for row in observations))
        self.assertTrue(any("jp_second_season" in row.slot_names for row in observations))

    def test_conservative_link_buckets(self) -> None:
        stats = build_corpus_stats(self._rows())
        score = score_candidate("电锯人", stats)
        self.assertEqual(
            link_bucket(
                "电锯人",
                score,
                BucketEvidence(self_anchor_concentration=0.8, official_alias_exclusive=True),
            ),
            LinkBucket.A_SELF,
        )
        self.assertEqual(
            link_bucket("电锯人", score, BucketEvidence(cross_work_hits=1)),
            LinkBucket.B_CROSS_WORK,
        )


if __name__ == "__main__":
    unittest.main()

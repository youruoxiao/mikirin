import unittest
from pathlib import Path

from mikirin.resolve_rules import (
    CandidateFeatures,
    RuleDecision,
    decide_all,
    decide_pair,
    evaluate,
    normalize,
    _parse_birthday,
    _parse_infobox,
)


def make_features(**overrides) -> CandidateFeatures:
    feat = CandidateFeatures(
        candidate_id="c1",
        left_entity_id="uuid5:character:left",
        right_entity_id="bgm:character:1",
        right_collects=10,          # 默认非路人，避免触发丢弃规则
        right_has_summary=True,
    )
    for key, value in overrides.items():
        setattr(feat, key, value)
    return feat


class NormalizeTests(unittest.TestCase):
    def test_strips_quotes_and_unifies_width(self):
        self.assertEqual(normalize('"加特"'), "加特")
        self.assertEqual(normalize("ＡＢＣ　１２３"), "abc123")
        self.assertEqual(normalize("孤独摇滚！"), normalize("孤独摇滚!"))

    def test_none_is_empty(self):
        self.assertEqual(normalize(None), "")


class FieldParsingTests(unittest.TestCase):
    def test_moegirl_birthday_list(self):
        self.assertEqual(_parse_birthday([None, 5, 11]), (5, 11))
        self.assertIsNone(_parse_birthday([None, 13, 40]))

    def test_infobox_birthday_text(self):
        self.assertEqual(_parse_birthday("5月11日"), (5, 11))

    def test_infobox_top_level_fields(self):
        fields = _parse_infobox("{{Infobox Crt\n|性别= 女\n|生日= 5月11日\n}}")
        self.assertEqual(fields["性别"], "女")
        self.assertEqual(fields["生日"], "5月11日")


class DecidePairTests(unittest.TestCase):
    def test_auto_same_with_work_name_and_corroboration(self):
        feat = make_features(
            left_names={"山田凉"}, right_names={"山田凉"},
            left_subjects={"孤独摇滚"}, right_subjects={"孤独摇滚"},
            left_gender="female", right_gender="female",
        )
        decision = decide_pair(feat)
        self.assertEqual(decision.decision, "auto_same")
        self.assertIn("same_work", decision.matched_evidence)
        self.assertIn("same_gender", decision.matched_evidence)

    def test_generic_single_char_name_needs_llm_without_corroboration(self):
        feat = make_features(
            left_names={"绯"}, right_names={"绯"},
            left_subjects={"某作品"}, right_subjects={"某作品"},
        )
        self.assertEqual(decide_pair(feat).decision, "needs_llm")

    def test_gender_conflict_same_work_is_gray_zone(self):
        # 投票制下同作品但性别冲突 = 2+2-2 = 2.0 分，落在灰区：
        # 可能是数据错误也可能是不同角色，不硬判
        feat = make_features(
            left_names={"某郎"}, right_names={"某郎"},
            left_subjects={"某作品"}, right_subjects={"某作品"},
            left_gender="male", right_gender="female",
        )
        decision = decide_pair(feat)
        self.assertEqual(decision.decision, "needs_llm")

    def test_no_work_overlap_without_conflict_needs_llm(self):
        feat = make_features(
            left_names={"某角色"}, right_names={"某角色"},
            left_subjects={"作品a"}, right_subjects={"作品b"},
        )
        self.assertEqual(decide_pair(feat).decision, "needs_llm")

    def test_name_gender_cv_triple_match_auto_same_without_work(self):
        feat = make_features(
            left_names={"山田凉"}, right_names={"山田凉"},
            left_subjects={"孤独摇滚"}, right_subjects=set(),
            left_gender="female", right_gender="female",
            left_voice_actors={"水野朔"}, right_voice_actors={"水野朔"},
        )
        decision = decide_pair(feat)
        self.assertEqual(decision.decision, "auto_same")
        self.assertIn("same_voice_actor", decision.matched_evidence)

    def test_gender_and_cv_double_conflict_auto_different(self):
        feat = make_features(
            left_names={"某角色"}, right_names={"某角色"},
            left_subjects={"作品a"}, right_subjects={"作品b"},
            left_gender="male", right_gender="female",
            left_voice_actors={"声优甲"}, right_voice_actors={"声优乙"},
        )
        decision = decide_pair(feat)
        self.assertEqual(decision.decision, "auto_different")
        self.assertIn("different_voice_actor", decision.conflicting_evidence)

    def test_fuzzy_work_match_auto_same(self):
        feat = make_features(
            left_names={"某角色"}, right_names={"某角色"},
            left_subjects={"合金弹头系列"}, right_subjects={"合金弹头3"},
            left_gender="female", right_gender="female",
        )
        decision = decide_pair(feat)
        self.assertEqual(decision.decision, "auto_same")
        self.assertIn("fuzzy_work", decision.matched_evidence)

    def test_bgm_nobody_is_dropped(self):
        feat = make_features(
            left_names={"某角色"}, right_names={"某角色"},
            left_subjects={"作品a"}, right_subjects={"作品b"},
            right_collects=0, right_has_summary=False,
        )
        decisions = {d.candidate_id: d for d in decide_all([feat])}
        self.assertEqual(decisions["c1"].decision, "dropped_sparse")


class GroupDecisionTests(unittest.TestCase):
    def test_single_work_winner_resolves_group(self):
        winner = make_features(
            candidate_id="c1",
            left_names={"歌野芋子"}, right_names={"歌野芋子"},
            left_subjects={"作品a"}, right_subjects={"作品a"},
            left_gender="female", right_gender="female",
        )
        loser = make_features(candidate_id="c2", right_entity_id="bgm:character:2",
                              left_names={"歌野芋子"}, right_names={"别的人"},
                              left_subjects={"作品a"}, right_subjects={"作品b"})
        loser.left_entity_id = winner.left_entity_id
        decisions = {d.candidate_id: d for d in decide_all([winner, loser])}
        self.assertEqual(decisions["c1"].decision, "auto_same")
        self.assertEqual(decisions["c2"].decision, "auto_different")

    def test_name_matching_loser_goes_to_llm_not_auto_different(self):
        # 衍生 incarnation（外传同角色）与正主同作品落选，规则不判，交 LLM
        winner = make_features(
            candidate_id="c1",
            left_names={"国木田花丸"}, right_names={"国木田花丸"},
            left_subjects={"lovelivesunshine"}, right_subjects={"lovelivesunshine"},
            left_gender="female", right_gender="female",
        )
        loser = make_features(candidate_id="c2", right_entity_id="bgm:character:2",
                              left_names={"国木田花丸"}, right_names={"国木田花丸"},
                              left_subjects={"lovelivesunshine"}, right_subjects={"幻日夜羽"})
        loser.left_entity_id = winner.left_entity_id
        decisions = {d.candidate_id: d for d in decide_all([winner, loser])}
        self.assertEqual(decisions["c1"].decision, "auto_same")
        self.assertEqual(decisions["c2"].decision, "needs_llm")

    def test_two_work_winners_stay_needs_llm(self):
        first = make_features(
            candidate_id="c1",
            left_names={"绯"}, right_names={"绯"},
            left_subjects={"作品a"}, right_subjects={"作品a"},
        )
        second = make_features(candidate_id="c2", right_entity_id="bgm:character:2",
                               left_names={"绯"}, right_names={"绯"},
                               left_subjects={"作品a"}, right_subjects={"作品a"})
        second.left_entity_id = first.left_entity_id
        decisions = {d.candidate_id: d for d in decide_all([first, second])}
        self.assertEqual(decisions["c1"].decision, "needs_llm")
        self.assertEqual(decisions["c2"].decision, "needs_llm")


class EvaluateTests(unittest.TestCase):
    def test_precision_and_coverage(self):
        decisions = [
            RuleDecision("c1", "auto_same", [], [], "r"),
            RuleDecision("c2", "auto_same", [], [], "r"),
            RuleDecision("c3", "auto_different", [], [], "r"),
            RuleDecision("c4", "needs_llm", [], [], "r"),
        ]
        labels = {"c1": "same", "c2": "different", "c3": "different", "c4": "same"}
        report = evaluate(decisions, labels)
        self.assertEqual(report["auto_same"]["precision"], 0.5)
        self.assertEqual(report["auto_different"]["precision"], 1.0)
        self.assertEqual(report["rule_coverage"], 0.75)


if __name__ == "__main__":
    unittest.main()

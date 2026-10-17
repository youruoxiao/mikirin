import unittest

from mikirin.resolution_apply import compile_resolution


def rule(candidate_id, decision):
    return {
        "candidate_id": candidate_id,
        "decision": decision,
        "matched_evidence": ["same_work"],
        "conflicting_evidence": [],
    }


def llm(candidate_id, decision, confidence=0.95, review_required=False, flags=None):
    return {
        "candidate_id": candidate_id,
        "decision": decision,
        "confidence": confidence,
        "review_required": review_required,
        "review_flags": flags or [],
        "matched_evidence": [],
        "conflicting_evidence": [],
        "reason": "r",
    }


class CompileResolutionTests(unittest.TestCase):
    def test_rule_decisions_pass_through(self):
        compiled, review, summary = compile_resolution(
            {"c1": rule("c1", "auto_same"), "c2": rule("c2", "auto_different"),
             "c3": rule("c3", "dropped_sparse")},
            {},
        )
        finals = {c["candidate_id"]: c["final_decision"] for c in compiled}
        self.assertEqual(finals, {"c1": "auto_same", "c2": "auto_different",
                                  "c3": "dropped_sparse"})
        self.assertEqual(review, [])
        self.assertEqual(summary["final_counts"]["auto_same"], 1)

    def test_llm_high_confidence_same_accepted(self):
        compiled, review, _ = compile_resolution(
            {"c1": rule("c1", "needs_llm")},
            {"c1": llm("c1", "same", 0.95)},
        )
        self.assertEqual(compiled[0]["final_decision"], "auto_same")
        self.assertEqual(compiled[0]["decision_source"], "llm")
        self.assertEqual(review, [])

    def test_llm_low_confidence_same_goes_to_review(self):
        compiled, review, _ = compile_resolution(
            {"c1": rule("c1", "needs_llm")},
            {"c1": llm("c1", "same", 0.5)},
        )
        self.assertEqual(compiled[0]["final_decision"], "review")
        self.assertEqual(len(review), 1)

    def test_llm_flagged_goes_to_review(self):
        compiled, review, _ = compile_resolution(
            {"c1": rule("c1", "needs_llm")},
            {"c1": llm("c1", "different", 0.9, review_required=True,
                       flags=["generic_name"])},
        )
        self.assertEqual(compiled[0]["final_decision"], "review")
        self.assertEqual(review[0]["review_flags"], ["generic_name"])

    def test_llm_different_accepted(self):
        compiled, review, _ = compile_resolution(
            {"c1": rule("c1", "needs_llm")},
            {"c1": llm("c1", "different", 0.9)},
        )
        self.assertEqual(compiled[0]["final_decision"], "auto_different")
        self.assertEqual(review, [])

    def test_rule_wins_conflict_and_records_review(self):
        compiled, review, summary = compile_resolution(
            {"c1": rule("c1", "auto_same")},
            {"c1": llm("c1", "different", 0.9)},
        )
        self.assertEqual(compiled[0]["final_decision"], "auto_same")
        self.assertEqual(review[0]["issue"], "rule_llm_conflict")
        self.assertEqual(summary["rule_llm_conflicts"], 1)

    def test_no_decision_goes_to_review(self):
        compiled, review, _ = compile_resolution(
            {"c1": rule("c1", "needs_llm")}, {},
        )
        self.assertEqual(compiled[0]["final_decision"], "review")
        self.assertEqual(review[0]["issue"], "no_decision")


if __name__ == "__main__":
    unittest.main()

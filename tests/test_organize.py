import unittest

from mikirin.organize import validate_decisions


class OrganizerValidationTests(unittest.TestCase):
    def test_valid_decisions_are_normalized(self) -> None:
        result = validate_decisions({"decisions": [{
            "candidate_id": "candidate:1",
            "decision": "uncertain",
            "confidence": 0.4,
            "matched_evidence": ["same_full_name"],
            "conflicting_evidence": ["insufficient_work_evidence"],
            "reason": "证据不足",
        }]}, ["candidate:1"])
        self.assertEqual(result["decisions"][0]["decision"], "uncertain")

    def test_missing_or_extra_ids_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_decisions({"decisions": []}, ["candidate:1"])
        with self.assertRaises(ValueError):
            validate_decisions({"decisions": [{
                "candidate_id": "candidate:other",
                "decision": "same",
                "confidence": 1,
                "matched_evidence": ["same_work", "same_full_name"],
                "conflicting_evidence": [],
                "reason": "x",
            }]}, ["candidate:1"])

    def test_same_requires_work_and_independent_evidence(self) -> None:
        with self.assertRaises(ValueError):
            validate_decisions({"decisions": [{
                "candidate_id": "candidate:1",
                "decision": "same",
                "confidence": 0.95,
                "matched_evidence": ["same_full_name"],
                "conflicting_evidence": [],
                "reason": "只有同名",
            }]}, ["candidate:1"])

    def test_unknown_evidence_code_is_quarantined_not_lost(self) -> None:
        result = validate_decisions({"decisions": [{
            "candidate_id": "candidate:1",
            "decision": "different",
            "confidence": 0.8,
            "matched_evidence": [],
            "conflicting_evidence": ["different_subjects"],
            "reason": "作品不同",
        }]}, ["candidate:1"])
        decision = result["decisions"][0]
        self.assertEqual(decision["conflicting_evidence"], ["other_conflict"])
        self.assertEqual(decision["unrecognized_evidence"], ["different_subjects"])


if __name__ == "__main__":
    unittest.main()

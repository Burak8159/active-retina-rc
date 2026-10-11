import unittest

from active_retina.audit_validation_checkpoint_ceiling import evaluate_rows


class ValidationCheckpointCeilingTests(unittest.TestCase):
    def test_ungradable_routing_uses_model_quality_score(self):
        rows = [
            {"split": "valid", "eye_id": "n0", "grade": "0", "quality_known": "1", "ungradable": "0", "risk": "0.1", "quality": "0.1"},
            {"split": "valid", "eye_id": "n1", "grade": "1", "quality_known": "1", "ungradable": "0", "risk": "0.2", "quality": "0.1"},
            {"split": "valid", "eye_id": "p0", "grade": "2", "quality_known": "1", "ungradable": "0", "risk": "0.8", "quality": "0.1"},
            {"split": "valid", "eye_id": "p1", "grade": "3", "quality_known": "1", "ungradable": "0", "risk": "0.9", "quality": "0.1"},
            {"split": "valid", "eye_id": "u0", "grade": "5", "quality_known": "1", "ungradable": "1", "risk": "0.1", "quality": "0.9"},
        ]

        result = evaluate_rows(rows, "risk")

        self.assertEqual(result["metrics"]["missed_referable_dr"], 0)
        self.assertEqual(result["metrics"]["missed_known_ungradable"], 0)
        self.assertEqual(result["metrics"]["case_action_coverage"], 1.0)

    def test_exclusion_removes_only_named_validation_eye(self):
        rows = [
            {"split": "valid", "eye_id": "n0", "grade": "0", "quality_known": "0", "ungradable": "0", "risk": "0.1", "quality": "0.1"},
            {"split": "valid", "eye_id": "n1", "grade": "1", "quality_known": "0", "ungradable": "0", "risk": "0.2", "quality": "0.1"},
            {"split": "valid", "eye_id": "p0", "grade": "2", "quality_known": "0", "ungradable": "0", "risk": "0.8", "quality": "0.1"},
            {"split": "test", "eye_id": "p1", "grade": "2", "quality_known": "0", "ungradable": "0", "risk": "0.9", "quality": "0.1"},
        ]

        result = evaluate_rows(rows, "risk", {"n1"})

        self.assertEqual(result["metrics"]["n"], 2)
        self.assertEqual(result["metrics"]["gradable_nonreferable"], 1)


if __name__ == "__main__":
    unittest.main()

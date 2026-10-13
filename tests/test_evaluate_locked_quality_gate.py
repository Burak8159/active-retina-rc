import unittest

from active_retina.evaluate_locked_quality_gate import summarize


class LockedQualityGateEvaluationTests(unittest.TestCase):
    def test_uses_fixed_thresholds_and_reports_baseline_union_without_label_tuning(self):
        reference = [
            {"eye_id": "n1", "source": "DDR", "grade": "0", "referable": "0", "ungradable": "0", "action": "REFER"},
            {"eye_id": "n2", "source": "DDR", "grade": "1", "referable": "0", "ungradable": "0", "action": "CLEAR"},
            {"eye_id": "p1", "source": "DDR", "grade": "2", "referable": "1", "ungradable": "0", "action": "REFER"},
            {"eye_id": "u1", "source": "DDR", "grade": "5", "referable": "0", "ungradable": "1", "action": "REFER"},
        ]
        manifest = [
            {"image_id": eye, "eye_id": eye, "source_dataset": "DDR"}
            for eye in ("n1", "n2", "p1", "u1")
        ]
        scores = {
            "n1": (-3.0, -5.0),
            "n2": (-3.0, -5.0),
            "p1": (1.0, -5.0),
            "u1": (-4.0, 0.0),
        }

        predictions, result = summarize(reference, manifest, scores, -2.0, -1.0)

        self.assertEqual(len(predictions), 4)
        self.assertEqual(result["status"], "EXPOSED_REFERENCE_TRANSFER_DIAGNOSTIC_NOT_CERTIFICATION")
        self.assertEqual(result["candidate_standalone"]["missed_referable"], 0)
        self.assertEqual(result["candidate_standalone"]["missed_ungradable"], 0)
        self.assertEqual(result["frozen_baseline_plus_candidate"]["additional_true_negatives"], 1)
        self.assertEqual(result["frozen_baseline_plus_candidate"]["gap_to_75_percent"], 0)


if __name__ == "__main__":
    unittest.main()

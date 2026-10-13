import unittest

import numpy as np

from active_retina.train_mmrdr_candidate import select_quality_gated_policy


class QualityGatePolicyTests(unittest.TestCase):
    def test_separate_risk_and_ungradable_thresholds_preserve_full_action_coverage(self):
        grades = np.asarray([0, 1, 2, 3, 5, 5])
        risk = np.asarray([-3.0, 0.2, 0.1, 0.4, -4.0, 0.0])
        quality = np.asarray([-2.0, -2.0, -2.0, -2.0, 0.5, 1.0])

        result = select_quality_gated_policy(grades, risk, quality)

        self.assertEqual(result["false_negative_clearance"], 0)
        self.assertEqual(result["missed_ungradable"], 0)
        self.assertEqual(result["true_negative"], 1)
        self.assertEqual(result["negative"], 2)
        self.assertEqual(result["action_coverage"], 1.0)
        self.assertEqual(result["risk_threshold"], 0.1)
        self.assertEqual(result["quality_threshold"], 0.5)

    def test_paired_views_are_aggregated_to_eye_before_threshold_selection(self):
        grades = np.asarray([0, 0, 2, 3, 5, 5])
        risk = np.asarray([-3.0, 0.2, 0.1, 0.4, -4.0, 0.0])
        quality = np.asarray([-2.0, -2.0, -2.0, -2.0, 0.5, 1.0])
        groups = ["n1", "n2", "positive", "positive", "ungradable", "ungradable"]

        result = select_quality_gated_policy(grades, risk, quality, groups)

        self.assertEqual(result["risk_threshold"], 0.4)
        self.assertEqual(result["negative"], 2)
        self.assertEqual(result["true_negative"], 2)
        self.assertEqual(result["false_negative_clearance"], 0)
        self.assertEqual(result["missed_ungradable"], 0)


if __name__ == "__main__":
    unittest.main()

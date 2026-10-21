import unittest

import pandas as pd

from paper.reproducibility.paired_replay_crossfit_audit import paired_patient_bootstrap


class PairedReplayCrossfitAuditTests(unittest.TestCase):
    def test_bootstrap_keeps_paired_patient_clusters_and_is_seeded(self):
        rows = pd.DataFrame([
            {"source": "A", "patient_id": "p1", "eye_id": "e1", "referable": 0,
             "ungradable": 0, "replay_clear": 1, "crossfit_clear": 1},
            {"source": "A", "patient_id": "p1", "eye_id": "e2", "referable": 0,
             "ungradable": 0, "replay_clear": 1, "crossfit_clear": 0},
            {"source": "A", "patient_id": "p2", "eye_id": "e3", "referable": 0,
             "ungradable": 0, "replay_clear": 0, "crossfit_clear": 0},
        ])
        first = paired_patient_bootstrap(rows, replicates=500, seed=41)
        second = paired_patient_bootstrap(rows, replicates=500, seed=41)
        self.assertEqual(first, second)
        self.assertEqual(first["negative_patient_clusters"], 2)
        self.assertEqual(first["eligible_negative_eyes"], 3)
        self.assertEqual(first["replay_true_negatives"], 2)
        self.assertEqual(first["crossfit_true_negatives"], 1)
        self.assertAlmostEqual(first["difference_percentage_points"], -100 / 3)
        self.assertEqual(first["required_case_miss_comparison"]["required_case_eyes"], 0)

    def test_empty_eligible_negative_subset_fails_closed(self):
        rows = pd.DataFrame([
            {"source": "A", "patient_id": "p1", "eye_id": "e1", "referable": 1,
             "ungradable": 0, "replay_clear": 0, "crossfit_clear": 0},
        ])
        with self.assertRaisesRegex(ValueError, "no gradable nonreferable eyes"):
            paired_patient_bootstrap(rows, replicates=10)

    def test_paired_missed_case_risk_uses_patient_clusters(self):
        rows = pd.DataFrame([
            {"source": "A", "patient_id": "p1", "eye_id": "e1", "referable": 1,
             "ungradable": 0, "replay_clear": 0, "crossfit_clear": 1},
            {"source": "A", "patient_id": "p1", "eye_id": "e2", "referable": 1,
             "ungradable": 0, "replay_clear": 0, "crossfit_clear": 0},
            {"source": "A", "patient_id": "p2", "eye_id": "e3", "referable": 0,
             "ungradable": 0, "replay_clear": 1, "crossfit_clear": 0},
        ])
        result = paired_patient_bootstrap(rows, replicates=500, seed=7)
        miss = result["required_case_miss_comparison"]
        self.assertEqual(miss["required_case_eyes"], 2)
        self.assertEqual(miss["required_case_patient_clusters"], 1)
        self.assertEqual(miss["replay_misses"], 0)
        self.assertEqual(miss["crossfit_misses"], 1)
        self.assertEqual(miss["difference_percentage_points"], 50.0)
        self.assertEqual(miss["patient_cluster_bootstrap"]["difference_95_percentile_interval_pp"], [50.0, 50.0])


if __name__ == "__main__":
    unittest.main()

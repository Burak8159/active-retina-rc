import unittest

from active_retina.evaluate_grouped_accuracy_threshold import (
    accuracy_selected_threshold,
    grouped_folds,
)


class GroupedAccuracyThresholdTests(unittest.TestCase):
    def test_threshold_maximizes_accuracy_on_training_indices(self):
        rows = [
            {"patient_id": "p1", "label": 0, "score": 0.1},
            {"patient_id": "p2", "label": 0, "score": 0.2},
            {"patient_id": "p3", "label": 1, "score": 0.8},
            {"patient_id": "p4", "label": 1, "score": 0.9},
        ]
        self.assertAlmostEqual(accuracy_selected_threshold(rows, list(range(4))), 0.5)

    def test_patient_groups_never_cross_folds(self):
        rows = [
            {"patient_id": patient, "label": label, "score": float(index)}
            for index, (patient, label) in enumerate(
                [("a", 0), ("a", 1), ("b", 0), ("c", 1), ("d", 0), ("e", 1)]
            )
        ]
        folds = grouped_folds(rows, 3)
        memberships = {}
        for fold_index, fold in enumerate(folds):
            for row_index in fold:
                patient = rows[row_index]["patient_id"]
                memberships.setdefault(patient, set()).add(fold_index)
        self.assertEqual(len(memberships), 5)
        self.assertTrue(all(len(fold_ids) == 1 for fold_ids in memberships.values()))


if __name__ == "__main__":
    unittest.main()

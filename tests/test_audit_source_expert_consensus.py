import unittest

from active_retina.audit_source_expert_consensus import _stratified_group_folds


class SourceExpertConsensusTests(unittest.TestCase):
    def test_group_folds_are_deterministic_and_keep_patient_eyes_together(self):
        rows = []
        for patient in range(20):
            grade = 0 if patient % 2 == 0 else 2
            for eye in ("L", "R"):
                rows.append({
                    "patient_id": f"p{patient}",
                    "source_dataset": "source_a",
                    "grade": str(grade),
                })

        first = _stratified_group_folds(rows, 5)
        second = _stratified_group_folds(rows, 5)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(set(first.values()), set(range(5)))

    def test_sources_are_not_collapsed_when_patient_ids_overlap(self):
        rows = []
        for source in ("source_a", "source_b"):
            for patient in range(10):
                rows.append({
                    "patient_id": f"patient-{patient}",
                    "source_dataset": source,
                    "grade": "0" if patient % 2 == 0 else "2",
                })

        folds = _stratified_group_folds(rows, 5)

        self.assertEqual(len(folds), 20)
        self.assertIn("source_a\0patient-0", folds)
        self.assertIn("source_b\0patient-0", folds)


if __name__ == "__main__":
    unittest.main()

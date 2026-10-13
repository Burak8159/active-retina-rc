import unittest

from active_retina.train_mmrdr_candidate import source_patient_sampling_weights


class SourcePatientSamplingTests(unittest.TestCase):
    def test_equalizes_source_and_patient_mass_without_overweighting_paired_images(self):
        rows = [
            {"source_dataset": "A", "patient_id": "p1"},
            {"source_dataset": "A", "patient_id": "p1"},
            {"source_dataset": "A", "patient_id": "p2"},
            {"source_dataset": "B", "patient_id": "p3"},
        ]
        weights = source_patient_sampling_weights(rows)
        self.assertAlmostEqual(sum(weights[:3]), 0.5)
        self.assertAlmostEqual(sum(weights[3:]), 0.5)
        self.assertAlmostEqual(sum(weights[:2]), weights[2])

    def test_ddr_manifest_subsets_share_one_source_budget(self):
        rows = [
            {"source_dataset": "ddr_hf_specificity300", "patient_id": "p1"},
            {"source_dataset": "ddr_full_added", "patient_id": "p2"},
            {"source_dataset": "drtid_paired", "patient_id": "q1"},
        ]
        weights = source_patient_sampling_weights(rows)
        self.assertAlmostEqual(weights[0] + weights[1], 0.5)
        self.assertAlmostEqual(weights[2], 0.5)


if __name__ == "__main__":
    unittest.main()

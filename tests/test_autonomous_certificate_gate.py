import unittest

from active_retina.certify_autonomous_candidate import patient_certificate


def make_patient(patient_id, positive, predictions):
    return [
        {
            "patient_id": patient_id,
            "eye_id": f"{patient_id}-eye-{index}",
            "referable_dr": int(positive),
            "prediction": prediction,
        }
        for index, prediction in enumerate(predictions)
    ]


class AutonomousCertificateGateTests(unittest.TestCase):
    def test_requires_full_binary_coverage_in_addition_to_risk_certificate(self):
        rows = []
        for index in range(59):
            rows.extend(make_patient(f"positive-{index}", True, [1]))
        rows.extend(make_patient("negative-1", False, [0]))
        rows.extend(make_patient("negative-2", False, [""]))

        result = patient_certificate(rows)

        self.assertTrue(result["certifies_5_percent_95_confidence"])
        self.assertFalse(result["full_binary_eye_row_coverage"])
        self.assertFalse(result["meets_full_binary_80_0_100_research_target"])

    def test_requires_zero_missed_positive_eyes_even_if_patient_was_referred(self):
        rows = []
        rows.extend(make_patient("positive-1", True, [0, 1]))
        for index in range(59):
            rows.extend(make_patient(f"positive-{index + 2}", True, [1]))
        rows.extend(make_patient("negative-1", False, [0]))

        result = patient_certificate(rows)

        self.assertEqual(result["missed_positive_patients"], 0)
        self.assertEqual(result["missed_positive_eye_rows"], 1)
        self.assertFalse(result["meets_full_binary_80_0_100_research_target"])

    def test_passes_research_gate_only_when_all_criteria_are_met(self):
        rows = []
        for index in range(59):
            rows.extend(make_patient(f"positive-{index}", True, [1]))
        for index in range(20):
            rows.extend(make_patient(f"negative-{index}", False, [0]))

        result = patient_certificate(rows)

        self.assertTrue(result["full_binary_eye_row_coverage"])
        self.assertEqual(result["missed_positive_patients"], 0)
        self.assertEqual(result["missed_positive_eye_rows"], 0)
        self.assertEqual(result["specificity_decided_negative_patients"], 1.0)
        self.assertTrue(result["certifies_5_percent_95_confidence"])
        self.assertTrue(result["meets_full_binary_80_0_100_research_target"])


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from active_retina.clinical_case_runner import classify_forced_binary, html_report
from active_retina.clinical_station import assessment_summary, form_page, public_scan_response


class FixedBinary:
    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, x):
        return np.asarray([[1.0 - self.probability, self.probability]])


class FixedGrade:
    def predict(self, x):
        return np.asarray([1])


class ClinicalStationUiTests(unittest.TestCase):
    def test_scan_form_is_minimal_and_hides_research_baselines(self):
        page = form_page().decode("utf-8")
        self.assertIn('name="first_image"', page)
        self.assertIn('name="second_image"', page)
        self.assertIn("Analyze scan", page)
        self.assertIn('name="forced_binary"', page)
        self.assertIn("not clinically validated", page)
        self.assertNotIn("70.13", page)
        self.assertNotIn("63.12", page)
        self.assertNotIn("operator_id", page)
        self.assertNotIn("Case ID", page)

    def test_result_language_is_simple_and_preserves_safe_next_steps(self):
        self.assertEqual(
            assessment_summary({"recommendation": "manual_review_likely_referable"}),
            {"result": "Possible referable diabetic retinopathy", "next": "Arrange clinician assessment."},
        )
        self.assertEqual(
            assessment_summary({"recommendation": "second_view_required_before_recommendation"}),
            {"result": "A second retinal image is needed", "next": "Capture the other retinal view, then review again."},
        )
        self.assertEqual(
            assessment_summary({"recommendation": "manual_review_image_quality_failed"}),
            {"result": "Image quality is not sufficient", "next": "Capture a clearer retinal image."},
        )

    def test_public_api_response_contains_only_scan_result_fields(self):
        response = public_scan_response({
            "case_id": "CASE-1",
            "decision": {"recommendation": "manual_review_likely_referable", "referable_score": 0.92},
        }, "/case?id=CASE-1")
        self.assertEqual(set(response), {
            "case_id", "assessment", "next_step", "clinician_confirmation_required",
            "autonomous_diagnosis", "report_url",
        })
        self.assertEqual(response["assessment"], "Possible referable diabetic retinopathy")
        self.assertTrue(response["clinician_confirmation_required"])
        self.assertFalse(response["autonomous_diagnosis"])

    def test_forced_binary_response_is_explicitly_unvalidated_and_not_7013(self):
        response = public_scan_response({
            "case_id": "CASE-2",
            "decision": {
                "action": "CLEAR",
                "recommendation": "forced_binary_clear_research_output",
                "binary_mode": "forced_binary_research_unvalidated",
                "binary_threshold": 0.5,
                "referable_score": 0.2,
            },
        }, "/case?id=CASE-2")
        self.assertEqual(response["binary_output"], "CLEAR")
        self.assertFalse(response["is_7013_policy"])
        self.assertFalse(response["autonomous_diagnosis"])

    def test_forced_binary_classifier_returns_action_for_single_and_paired_images(self):
        artifact = {"models": {
            "first_binary": FixedBinary(0.2),
            "first_grade": FixedGrade(),
            "paired_binary": FixedBinary(0.8),
            "paired_grade": FixedGrade(),
        }}
        image = np.zeros(4, dtype=np.float32)
        self.assertEqual(classify_forced_binary(artifact, image, None, True)["action"], "CLEAR")
        pair = classify_forced_binary(artifact, image, image, True)
        self.assertEqual(pair["action"], "REFER")
        self.assertEqual(pair["images_used"], 2)

    def test_forced_binary_quality_failure_routes_to_refer_and_report_renders(self):
        decision = classify_forced_binary({}, np.zeros(4), None, False)
        self.assertEqual(decision["action"], "REFER")
        report = html_report({"case_id": "q1", "decision": decision, "model_version": "test", "model_status": "research"})
        self.assertIn("quality failure routed to REFER", report)
        self.assertIn("REFER", report)


if __name__ == "__main__":
    unittest.main()

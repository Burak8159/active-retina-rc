import unittest

from active_retina.clinical_station import assessment_summary, form_page, public_scan_response


class ClinicalStationUiTests(unittest.TestCase):
    def test_scan_form_is_minimal_and_hides_research_baselines(self):
        page = form_page().decode("utf-8")
        self.assertIn('name="first_image"', page)
        self.assertIn('name="second_image"', page)
        self.assertIn("Analyze scan", page)
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


if __name__ == "__main__":
    unittest.main()

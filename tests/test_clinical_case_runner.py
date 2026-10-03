from active_retina.clinical_case_runner import enrich_decision, html_report


def test_enrich_decision_adds_clinician_facing_analysis():
    decision = enrich_decision(
        {
            "action": "STOP_AFTER_FIRST_VIEW",
            "recommendation": "manual_review_low_referable_risk",
            "referable_score": 0.01,
            "grade_prediction": 0,
            "images_used": 1,
            "needs_second_view": False,
        }
    )
    assert decision["referable_probability"] == 0.01
    assert decision["predicted_dr_grade"] == 0
    assert decision["model_impression_for_clinician_review"] == "low_referable_dr_probability"
    assert decision["recommended_next_step"] == "clinician_review_required_before_final_decision"


def test_html_report_marks_manual_review_required():
    report = html_report(
        {
            "case_id": "demo",
            "model_version": "version",
            "model_status": "status",
            "decision": {
                "action": "STOP_AFTER_FIRST_VIEW",
                "recommendation": "manual_review_low_referable_risk",
                "referable_score": 0.01,
                "grade_prediction": 0,
                "model_impression_for_clinician_review": "low_referable_dr_probability",
                "recommended_next_step": "clinician_review_required_before_final_decision",
                "images_used": 1,
            },
        }
    )
    assert "Manual review required" in report
    assert "STOP_AFTER_FIRST_VIEW" in report
    assert "autonomous diagnosis" in report

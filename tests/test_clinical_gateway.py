from active_retina.clinical_gateway import GatewayInput, decide


def test_gateway_blocks_clinical_use():
    out = decide(
        GatewayInput(
            schema_version="1.0",
            case_id="case",
            operator_id="op",
            first_view_score=0.01,
            anatomy_mapping_verified=True,
            intended_use="clinical",
        )
    )
    assert out.action == "DEFER_TO_CLINICIAN"
    assert out.must_review is True
    assert out.clinically_actionable is False


def test_gateway_blocks_unverified_anatomy_mapping():
    out = decide(
        GatewayInput(
            schema_version="1.0",
            case_id="case",
            operator_id="op",
            first_view_score=0.01,
            anatomy_mapping_verified=False,
            intended_use="research",
        )
    )
    assert out.action == "DEFER_TO_CLINICIAN"
    assert out.clinically_actionable is False


def test_gateway_research_second_view_when_uncertain_and_mapping_verified():
    out = decide(
        GatewayInput(
            schema_version="1.0",
            case_id="case",
            operator_id="op",
            first_view_score=0.42,
            anatomy_mapping_verified=True,
            intended_use="research",
        )
    )
    assert out.action == "ACQUIRE_SECOND_VIEW"
    assert out.images_required == 2
    assert out.must_review is True
    assert out.clinically_actionable is False

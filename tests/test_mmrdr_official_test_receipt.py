import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_roc_figure_supplement_and_markers_match_frozen_run_card():
    run_card_path = ROOT / "paper/reproducibility/mmrdr_uwf_official_test_run_card.json"
    run_card = json.loads(run_card_path.read_text(encoding="utf-8"))
    diagnostic_meta = run_card["posthoc_score_diagnostic"]
    diagnostic_path = ROOT / diagnostic_meta["aggregate_output"]
    roc_path = ROOT / diagnostic_meta["roc_display_csv"]
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    roc_rows = list(csv.DictReader(roc_path.open(encoding="utf-8", newline="")))

    assert hashlib.sha256(diagnostic_path.read_bytes()).hexdigest() == diagnostic_meta["aggregate_output_sha256"]
    assert hashlib.sha256(roc_path.read_bytes()).hexdigest() == diagnostic_meta["roc_display_csv_sha256"]
    assert diagnostic["export_sha256"] == diagnostic_meta["prediction_export_sha256"]
    assert diagnostic["n"] == run_card["dataset"]["official_test_rows"] == 2597
    assert diagnostic["roc_auc"] == diagnostic_meta["roc_auc_image_level"]
    assert diagnostic["frozen_operating_point"] == {
        key: run_card["official_test_result"][key]
        for key in ("true_positive", "false_negative", "true_negative", "false_positive", "sensitivity", "specificity")
    } | {"referral_rate": diagnostic["frozen_operating_point"]["referral_rate"]}
    assert len(roc_rows) == diagnostic_meta["roc_display_grid_points"] + 1
    assert [float(roc_rows[0]["false_positive_rate"]), float(roc_rows[0]["true_positive_rate"])] == [0.0, 0.0]
    assert [float(roc_rows[-1]["false_positive_rate"]), float(roc_rows[-1]["true_positive_rate"])] == [1.0, 1.0]
    assert all(
        float(left["false_positive_rate"]) <= float(right["false_positive_rate"])
        and float(left["true_positive_rate"]) <= float(right["true_positive_rate"])
        for left, right in zip(roc_rows, roc_rows[1:])
    )
    assert [
        {key: float(row[key]) for key in ("false_positive_rate", "true_positive_rate")}
        for row in roc_rows[1:]
    ] == diagnostic["roc_display_grid"]

    locked = run_card["official_test_result"]
    zero_miss = diagnostic["sensitivity_specificity_frontier"][0]
    expected_markers = (
        f"({1 - locked['specificity']:.6f},{locked['sensitivity']:.6f})",
        f"({1 - zero_miss['specificity']:.6f},{zero_miss['sensitivity']:.1f})",
    )
    for manuscript in ("paper/active_retina_ieee_so_far.tex", "paper/active_retina_arxiv_full.tex"):
        source = (ROOT / manuscript).read_text(encoding="utf-8")
        figure = source.split(r"\label{fig:mmrdr-roc}", 1)[0].rsplit(r"\begin{figure}", 1)[-1]
        assert all(marker in figure for marker in expected_markers)
        assert "post-hoc" in figure.lower()


def test_local_mmrdr_export_audit_receipt_matches_run_card():
    receipt = json.loads(
        (ROOT / "paper/reproducibility/local_mmrdr_prediction_export_audit_20261019.json")
        .read_text(encoding="utf-8")
    )
    run_card = json.loads(
        (ROOT / "paper/reproducibility/mmrdr_uwf_official_test_run_card.json")
        .read_text(encoding="utf-8")
    )

    assert receipt["status"] == "PASS_LOCAL_PREDICTION_EXPORT_AGGREGATE_AUDIT_NOT_INFERENCE_REPRODUCTION"
    assert receipt["scope"].endswith("does not reproduce image inference or establish clinical validation")
    assert receipt["export_sha256"] == run_card["provenance_hashes"]["test_prediction_export_local_only"]["sha256"]
    assert receipt["rows"] == run_card["dataset"]["official_test_rows"] == 2597
    assert receipt["confusion"] == {
        key: run_card["official_test_result"][key]
        for key in ("true_positive", "false_negative", "true_negative", "false_positive")
    }
    assert receipt["grade_actions"] == run_card["official_test_grade_actions"]
    assert receipt["sensitivity_wilson_95ci"] == run_card["official_test_result"]["sensitivity_wilson_95ci"]
    assert receipt["specificity_wilson_95ci"] == run_card["official_test_result"]["specificity_wilson_95ci"]
    assert "image_id" not in receipt and "patient_id" not in receipt


def test_run_card_records_grade_loss_semantics_and_unverified_run_coefficient():
    run_card = json.loads(
        (ROOT / "paper/reproducibility/mmrdr_uwf_official_test_run_card.json")
        .read_text(encoding="utf-8")
    )
    semantics = run_card["reconstructed_training_provenance"]["loss_semantics"]
    assert semantics["grade_term_option_name"] == "ordinal_weight"
    assert semantics["grade_term_is_order_aware"] is False
    assert semantics["grade_term_implementation"] == (
        "inverse-frequency weighted torch.nn.CrossEntropyLoss over the integer grade class index"
    )
    assert semantics["checked_in_default_coefficient"] == 0.25
    assert semantics["run_specific_coefficient_verified"] is False
    assert semantics["observed_mmrdr_training_grade_indices"] == [0, 1, 2, 3, 4]
    assert semantics["grade_index_5_has_mmrdr_training_examples"] is False
    assert semantics["referable_logit_includes_index_5"] is True

    provenance = run_card["reconstructed_training_provenance"]
    source_files = provenance["source_commit_file_verification"]
    assert source_files["trainer_sha256_at_source_commit"] == provenance["trainer"]["sha256"]
    assert source_files["feature_extractor_sha256_at_source_commit"] == provenance["feature_extractor"]["sha256"]
    assert source_files["trainer_matches_recorded_sha256"] is True
    assert source_files["feature_extractor_matches_recorded_sha256"] is True
    assert source_files["actual_run_specific_arguments_recovered"] is False

    trainer_path = ROOT / "src/active_retina/train_mmrdr_cached_head.py"
    if trainer_path.is_file():
        trainer = trainer_path.read_text(encoding="utf-8")
        assert "nn.CrossEntropyLoss(weight=class_weights(" in trainer
        assert "args.ordinal_weight * criterion(logits, y_grade[idx])" in trainer


def test_development_trace_reconstruction_is_reported_as_non_test_diagnostic():
    run_card = json.loads(
        (ROOT / "paper/reproducibility/mmrdr_uwf_official_test_run_card.json")
        .read_text(encoding="utf-8")
    )
    diagnostic = run_card["default_recipe_reconstruction_check"]
    assert diagnostic["scope"] == (
        "Development manifest and cached features only; official test inputs and outputs were not read by this check."
    )
    assert diagnostic["historical_epoch_1"]["train_loss"] == 2.720549051332222
    assert diagnostic["historical_epoch_1"]["specificity_at_zero_miss"] == 0.04666666666666667
    assert diagnostic["historical_epoch_1"]["threshold"] == -1.350136160850525
    assert diagnostic["default_check_epoch_1"]["train_loss"] == 10.590698152497557
    assert diagnostic["default_check_epoch_1"]["specificity_at_zero_miss"] == 0.005
    assert diagnostic["default_check_epoch_1"]["threshold"] == 3.8017730712890625
    assert diagnostic["completed_validation_epochs_before_early_stop"] == 11
    assert diagnostic["stopped_before_checkpoint_save"] is True
    assert "not a new model result" in diagnostic["interpretation"]

    for relative in (
        "paper/active_retina_ieee_so_far.tex",
        "paper/active_retina_arxiv_full.tex",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "10.591" in source and "2.721" in source
        assert "did not read official-test inputs or outputs" in source
        assert "not a new model result" in source


def test_run_card_records_exact_join_but_not_patient_level_verification():
    run_card = json.loads(
        (ROOT / "paper/reproducibility/mmrdr_uwf_official_test_run_card.json")
        .read_text(encoding="utf-8")
    )
    dataset = run_card["dataset"]
    join = dataset["prediction_manifest_join"]
    source = dataset["source_annotation_csv"]
    join_receipt = json.loads(
        (ROOT / "paper/reproducibility/local_mmrdr_manifest_prediction_join_audit_20261019.json")
        .read_text(encoding="utf-8")
    )
    source_join = join_receipt["source_annotation_join"]

    assert dataset["test_manifest_patient_id_column_present"] is True
    assert dataset["test_manifest_patient_id_nonempty_rows"] == 0
    assert dataset["test_manifest_eye_id_column_present"] is True
    assert dataset["test_manifest_eye_id_nonempty_rows"] == 0
    assert join["exact_unique_image_id_set_match"] is True
    assert join["grade_and_referable_label_match_rows"] == 2597
    assert join["patient_clustered_uncertainty_available"] is False
    assert source["rows"] == 10404
    assert source["columns"] == ["type", "image", "grade", "lesion", "lr"]
    assert source["patient_link_key_column_present"] is False
    assert source["test_images_matched_by_image_name"] == 2597
    assert source["test_grade_matches"] == 2597
    assert source_join["source_annotations_sha256"] == source["sha256"]
    assert source_join["source_annotation_rows"] == source["rows"]
    assert source_join["test_manifest_rows_matched"] == 2597
    assert source_join["test_grade_match_count"] == 2597
    assert source_join["patient_level_link_key_available_in_source_annotations"] is False

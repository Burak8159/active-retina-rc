import csv
import hashlib

import pytest

from paper.reproducibility.audit_mmrdr_prediction_export import audit_export, wilson_interval


FIELDS = ["image_id", "grade", "referable_dr", "score", "threshold", "action"]
ROWS = [
    {"image_id": "synthetic-0", "grade": 0, "referable_dr": 0, "score": -0.5, "threshold": 0, "action": "CLEAR"},
    {"image_id": "synthetic-1", "grade": 1, "referable_dr": 0, "score": 0.5, "threshold": 0, "action": "REFER"},
    {"image_id": "synthetic-2", "grade": 2, "referable_dr": 1, "score": 0.5, "threshold": 0, "action": "REFER"},
    {"image_id": "synthetic-3", "grade": 3, "referable_dr": 1, "score": -0.5, "threshold": 0, "action": "CLEAR"},
    {"image_id": "synthetic-4", "grade": 4, "referable_dr": 1, "score": 0.5, "threshold": 0, "action": "REFER"},
]


def _write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _run_card(path, rows):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "provenance_hashes": {
            "test_prediction_export_local_only": {"sha256": digest}
        },
        "dataset": {"official_test_rows": len(rows)},
        "model_and_preprocessing": {"threshold": 0.0},
        "official_test_result": {
            "true_positive": 2,
            "false_negative": 1,
            "true_negative": 1,
            "false_positive": 1,
            "sensitivity_wilson_95ci": wilson_interval(2, 3),
            "specificity_wilson_95ci": wilson_interval(1, 2),
        },
        "official_test_grade_actions": [
            {"grade": 0, "images": 1, "referred": 0, "cleared": 1},
            {"grade": 1, "images": 1, "referred": 1, "cleared": 0},
            {"grade": 2, "images": 1, "referred": 1, "cleared": 0},
            {"grade": 3, "images": 1, "referred": 0, "cleared": 1},
            {"grade": 4, "images": 1, "referred": 1, "cleared": 0},
        ],
    }


def test_audits_aggregate_export_without_returning_row_ids(tmp_path):
    export = tmp_path / "synthetic_predictions.csv"
    _write_rows(export, ROWS)
    report = audit_export(export, _run_card(export, ROWS))

    assert report["status"] == "PASS_LOCAL_PREDICTION_EXPORT_AGGREGATE_AUDIT_NOT_INFERENCE_REPRODUCTION"
    assert report["rows"] == 5
    assert report["confusion"] == {
        "true_positive": 2,
        "false_negative": 1,
        "true_negative": 1,
        "false_positive": 1,
    }
    assert "synthetic-" not in str(report)


def test_rejects_export_when_action_disagrees_with_frozen_threshold(tmp_path):
    export = tmp_path / "synthetic_predictions.csv"
    corrupted = [dict(row) for row in ROWS]
    corrupted[0]["action"] = "REFER"
    _write_rows(export, corrupted)

    with pytest.raises(ValueError, match="stored action"):
        audit_export(export, _run_card(export, corrupted))


def test_rejects_label_that_conflicts_with_referable_grade(tmp_path):
    export = tmp_path / "synthetic_predictions.csv"
    corrupted = [dict(row) for row in ROWS]
    corrupted[2]["referable_dr"] = 0
    _write_rows(export, corrupted)

    with pytest.raises(ValueError, match="conflicts with grade"):
        audit_export(export, _run_card(export, corrupted))

import csv
import hashlib
from pathlib import Path

from active_retina.analyze_mmrdr_test_frontier import analyze


def test_score_ranking_and_posthoc_frontier_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    rows = [
        ["a", 0, 0, 0.1, 0.5, "CLEAR"],
        ["b", 0, 0, 0.4, 0.5, "CLEAR"],
        ["c", 2, 1, 0.6, 0.5, "REFER"],
        ["d", 3, 1, 0.9, 0.5, "REFER"],
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "grade", "referable_dr", "score", "threshold", "action"])
        writer.writerows(rows)

    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    result = analyze(path, expected_hash)

    assert result["roc_auc"] == 1.0
    assert result["average_precision"] == 1.0
    assert len(result["roc_display_grid"]) == 101
    assert result["roc_display_grid"][0] == {
        "false_positive_rate": 0.0,
        "true_positive_rate": 1.0,
    }
    assert result["roc_display_grid"][-1] == {
        "false_positive_rate": 1.0,
        "true_positive_rate": 1.0,
    }
    assert all(
        left["true_positive_rate"] <= right["true_positive_rate"]
        for left, right in zip(result["roc_display_grid"], result["roc_display_grid"][1:])
    )
    assert result["frozen_operating_point"] == {
        "true_positive": 2,
        "false_negative": 0,
        "true_negative": 2,
        "false_positive": 0,
        "sensitivity": 1.0,
        "specificity": 1.0,
        "referral_rate": 0.5,
    }
    assert result["sensitivity_specificity_frontier"][0]["minimum_sensitivity"] == 1.0


def test_frontier_refuses_modified_prediction_export(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    path.write_text(
        "grade,referable_dr,score,threshold,action\n0,0,0.1,0.5,CLEAR\n2,1,0.9,0.5,REFER\n",
        encoding="utf-8",
    )
    try:
        analyze(path, "0" * 64)
    except ValueError as exc:
        assert "hash" in str(exc)
    else:
        raise AssertionError("modified prediction export must fail closed")


def test_tied_scores_receive_average_auc_ranks(tmp_path: Path) -> None:
    path = tmp_path / "tied.csv"
    path.write_text(
        "grade,referable_dr,score,threshold,action\n"
        "0,0,0.5,0.5,REFER\n"
        "2,1,0.5,0.5,REFER\n",
        encoding="utf-8",
    )
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    result = analyze(path, expected_hash)
    assert result["roc_auc"] == 0.5
    assert result["average_precision"] == 0.5
    assert len(result["roc_display_grid"]) == 101

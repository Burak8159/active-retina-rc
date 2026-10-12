from __future__ import annotations

import numpy as np
import pytest

from active_retina.train_grade_boundary_head import _eye_rows, _metrics


def test_paired_feature_aggregation_is_view_order_invariant() -> None:
    rows = [
        {"source_split": "train", "eye_id": "e1", "image_id": "left", "grade": "1"},
        {"source_split": "train", "eye_id": "e1", "image_id": "right", "grade": "1"},
    ]
    features = {"left": np.array([1.0, 4.0]), "right": np.array([3.0, 2.0])}

    forward = _eye_rows(rows, features, "train")[0]["x"]
    reverse = _eye_rows(list(reversed(rows)), features, "train")[0]["x"]

    np.testing.assert_allclose(forward, reverse)
    np.testing.assert_allclose(forward, [2.0, 3.0, 2.0, 2.0, 1.0])


def test_single_view_aggregation_uses_zero_difference() -> None:
    rows = [{"source_split": "train", "eye_id": "e1", "image_id": "one", "grade": "0"}]

    result = _eye_rows(rows, {"one": np.array([2.0, 5.0])}, "train")[0]

    np.testing.assert_allclose(result["x"], [2.0, 5.0, 0.0, 0.0, 0.0])
    assert result["n_views"] == 1


def test_eye_with_conflicting_view_grades_is_rejected() -> None:
    rows = [
        {"source_split": "train", "eye_id": "e1", "image_id": "left", "grade": "1"},
        {"source_split": "train", "eye_id": "e1", "image_id": "right", "grade": "2"},
    ]

    with pytest.raises(ValueError, match="inconsistent eye grade"):
        _eye_rows(rows, {"left": np.ones(2), "right": np.zeros(2)}, "train")


def test_minimum_positive_cutoff_keeps_all_positive_cases_referred() -> None:
    rows = [
        {"grade": 0}, {"grade": 1}, {"grade": 2}, {"grade": 3},
    ]
    result = _metrics(rows, np.array([-1.0, -0.5, 0.2, 1.4]))

    assert result["true_negative"] == 2
    assert result["missed_referable"] == 0
    assert result["specificity_percent"] == 100.0
    assert result["binary_action_coverage_percent"] == 100.0

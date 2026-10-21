import csv

import pytest

from paper.reproducibility.audit_mmrdr_manifest_join import audit_manifest_join


def _write(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _inputs(tmp_path):
    manifest = tmp_path / "manifest.csv"
    predictions = tmp_path / "predictions.csv"
    _write(
        manifest,
        ["patient_id", "eye_id", "image_id", "image_path", "grade", "referable_dr"],
        [
            {"patient_id": "", "eye_id": "", "image_id": "img-a", "image_path": "images/a.jpg", "grade": "0", "referable_dr": "0"},
            {"patient_id": "", "eye_id": "", "image_id": "img-b", "image_path": "images/b.jpg", "grade": "2", "referable_dr": "1"},
        ],
    )
    _write(
        predictions,
        ["image_id", "grade", "referable_dr", "action"],
        [
            {"image_id": "img-a", "grade": "0", "referable_dr": "0", "action": "CLEAR"},
            {"image_id": "img-b", "grade": "2", "referable_dr": "1", "action": "REFER"},
        ],
    )
    return manifest, predictions


def test_exact_join_reports_blank_group_ids_without_emitting_them(tmp_path):
    manifest, predictions = _inputs(tmp_path)
    report = audit_manifest_join(manifest, predictions)

    assert report["exact_unique_image_id_set_match"] is True
    assert report["grade_match_count"] == report["referable_label_match_count"] == 2
    assert report["patient_id_column_present"] is True
    assert report["patient_id_nonempty_rows"] == 0
    assert report["eye_id_nonempty_rows"] == 0
    assert report["patient_clustered_uncertainty_available"] is False
    assert "img-" not in str(report)


@pytest.mark.parametrize("change, message", [
    ("grade", "grade mismatch"),
    ("label", "referable label mismatch"),
])
def test_rejects_joined_label_or_grade_mismatch(tmp_path, change, message):
    manifest, predictions = _inputs(tmp_path)
    rows = list(csv.DictReader(predictions.open(newline="", encoding="utf-8")))
    rows[1]["grade" if change == "grade" else "referable_dr"] = "1" if change == "grade" else "0"
    _write(predictions, ["image_id", "grade", "referable_dr", "action"], rows)

    with pytest.raises(ValueError, match=message):
        audit_manifest_join(manifest, predictions)


def test_rejects_nonidentical_image_id_sets(tmp_path):
    manifest, predictions = _inputs(tmp_path)
    rows = list(csv.DictReader(predictions.open(newline="", encoding="utf-8")))
    rows[1]["image_id"] = "different-image"
    _write(predictions, ["image_id", "grade", "referable_dr", "action"], rows)

    with pytest.raises(ValueError, match="sets do not match exactly"):
        audit_manifest_join(manifest, predictions)


def test_source_annotation_join_verifies_grades_but_not_patient_grouping(tmp_path):
    from paper.reproducibility.audit_mmrdr_manifest_join import audit_manifest_join

    manifest, predictions = _inputs(tmp_path)
    source = tmp_path / "source.csv"
    _write(
        source,
        ["type", "image", "grade", "lesion", "lr"],
        [
            {"type": "UWF", "image": "source/a.jpg", "grade": "0", "lesion": "[]", "lr": "0"},
            {"type": "UWF", "image": "source/b.jpg", "grade": "2", "lesion": "[]", "lr": "1"},
        ],
    )

    report = audit_manifest_join(manifest, predictions, source)
    source_audit = report["source_annotation_join"]
    assert source_audit["test_manifest_rows_matched"] == 2
    assert source_audit["test_grade_match_count"] == 2
    assert source_audit["source_patient_key_columns"] == []
    assert source_audit["source_laterality_column_present"] is True
    assert source_audit["patient_level_link_key_available_in_source_annotations"] is False
    assert "source/a.jpg" not in str(report)


def test_rejects_source_annotation_grade_mismatch(tmp_path):
    from paper.reproducibility.audit_mmrdr_manifest_join import audit_manifest_join

    manifest, predictions = _inputs(tmp_path)
    source = tmp_path / "source.csv"
    _write(
        source,
        ["image", "grade"],
        [{"image": "a.jpg", "grade": "0"}, {"image": "b.jpg", "grade": "1"}],
    )

    with pytest.raises(ValueError, match="grade mismatch against source annotations"):
        audit_manifest_join(manifest, predictions, source)

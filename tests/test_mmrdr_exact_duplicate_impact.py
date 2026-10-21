import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "paper/reproducibility/audit_mmrdr_exact_duplicate_impact.py"


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_exact_duplicate_audit_reports_aggregate_impact_without_ids(tmp_path):
    (tmp_path / "train_positive.jpg").write_bytes(b"duplicate-positive-fit")
    (tmp_path / "test_positive_fit_copy.jpg").write_bytes(b"duplicate-positive-fit")
    (tmp_path / "train_positive_valid.jpg").write_bytes(b"duplicate-positive-valid")
    (tmp_path / "test_positive_valid_copy.jpg").write_bytes(b"duplicate-positive-valid")
    (tmp_path / "test_negative.jpg").write_bytes(b"unique-negative")

    manifest = tmp_path / "manifest.csv"
    write_csv(
        manifest,
        ["image_id", "image_path", "grade", "source_split", "source_dataset"],
        [
            {"image_id": "fit_train", "image_path": str(tmp_path / "train_positive.jpg"), "grade": "2", "source_split": "train", "source_dataset": "mmrdr_uwf"},
            {"image_id": "validation_train", "image_path": str(tmp_path / "train_positive_valid.jpg"), "grade": "4", "source_split": "train", "source_dataset": "mmrdr_uwf"},
            {"image_id": "mmrdr_uwf_ts000001", "image_path": str(tmp_path / "test_positive_fit_copy.jpg"), "grade": "2", "source_split": "test", "source_dataset": "mmrdr_uwf"},
            {"image_id": "mmrdr_uwf_ts000002", "image_path": str(tmp_path / "test_positive_valid_copy.jpg"), "grade": "4", "source_split": "test", "source_dataset": "mmrdr_uwf"},
            {"image_id": "mmrdr_uwf_ts000003", "image_path": str(tmp_path / "test_negative.jpg"), "grade": "0", "source_split": "test", "source_dataset": "mmrdr_uwf"},
        ],
    )
    predictions = tmp_path / "predictions.csv"
    write_csv(
        predictions,
        ["image_id", "referable_dr", "action"],
        [
            {"image_id": "mmrdr_uwf_ts000001", "referable_dr": "1", "action": "REFER"},
            {"image_id": "mmrdr_uwf_ts000002", "referable_dr": "1", "action": "REFER"},
            {"image_id": "mmrdr_uwf_ts000003", "referable_dr": "0", "action": "CLEAR"},
        ],
    )
    cache = tmp_path / "features.npz"
    np.savez(
        cache,
        image_ids=np.asarray(["fit_train", "validation_train"]),
        splits=np.asarray(["train", "valid"]),
    )
    output = tmp_path / "aggregate.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest", str(manifest),
            "--prediction-export", str(predictions),
            "--feature-cache", str(cache),
            "--root", str(tmp_path),
            "--output", str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["status"] == "PASS_AGGREGATE_EXACT_BYTE_DUPLICATE_AUDIT"
    assert report["exact_duplicate_groups"] == 2
    assert report["groups_crossing_official_splits"] == 2
    assert report["test_duplicate_groups_with_exact_fit_copy"] == 1
    assert report["test_duplicate_groups_with_exact_selection_copy"] == 1
    assert report["test_duplicate_rows"] == 2
    assert report["test_duplicate_rows_positive"] == 2
    assert report["test_duplicate_rows_referred"] == 2
    assert report["official_test_metrics_all_rows"]["true_positive"] == 2
    assert report["official_test_metrics_excluding_exact_duplicate_images"]["true_positive"] == 0
    assert "image_ids" not in report and "duplicate_ids" not in report


def test_saved_overlap_receipt_is_consistent_with_both_manuscripts():
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads(
        (root / "paper/reproducibility/mmrdr_exact_duplicate_impact_20261019.json").read_text()
    )
    ieee = (root / "paper/active_retina_ieee_so_far.tex").read_text()
    arxiv = (root / "paper/active_retina_arxiv_full.tex").read_text()

    assert receipt["input_rows"] == 10404
    assert receipt["exact_duplicate_groups"] == 8
    assert receipt["groups_crossing_official_splits"] == 2
    assert receipt["test_duplicate_rows_positive"] == 2
    assert receipt["test_duplicate_rows_referred"] == 2
    assert receipt["official_test_metrics_excluding_exact_duplicate_images"]["false_negative"] == 5
    assert receipt["official_test_metrics_excluding_exact_duplicate_images"]["specificity_pct"] == receipt["official_test_metrics_all_rows"]["specificity_pct"]
    for source in (ieee, arxiv):
        assert "10,404" in source
        assert "two" in source and "model-fit data" in source
        assert "not wholly image-disjoint" in source or "not fully image-disjoint" in source
        assert "1,286" in source
        assert "20.02\\%" in source

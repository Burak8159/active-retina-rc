import csv

import numpy as np
from PIL import Image

from active_retina.train_rfmid_multilabel import multilabel_metrics, read_split


def test_read_split_joins_image_ids_and_binary_targets(tmp_path):
    image_dir = tmp_path / "Training_Set/Training_Set/Training"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(image_dir / "1.png")
    label_file = tmp_path / "Training_Set/Training_Set/RFMiD_Training_Labels.csv"
    label_file.parent.mkdir(parents=True, exist_ok=True)
    with label_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Disease_Risk", "ARMD"])
        writer.writerow(["1", "1", "1"])
    rows, labels = read_split(tmp_path, "train")
    assert labels == ["Disease_Risk", "ARMD"]
    assert rows[0]["id"] == "1"
    assert rows[0]["targets"] == [1, 1]
    assert rows[0]["image_path"].is_file()


def test_multilabel_metrics_keep_per_class_support_and_separate_metrics():
    y = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
    scores = np.array([[0.9, 0.2], [0.8, 0.7], [0.3, 0.1], [0.2, 0.6]])
    metrics = multilabel_metrics(y, scores, ["Disease_Risk", "ARMD"])
    assert metrics["n_images"] == 4
    assert metrics["per_label"]["ARMD"]["positive"] == 2
    assert metrics["per_label"]["ARMD"]["auroc"] == 1.0
    assert metrics["subset_accuracy_at_0_5"] == 1.0


def test_macro_metrics_exclude_labels_with_no_positive_training_support():
    y = np.array([[1, 1], [1, 1], [0, 0], [0, 0]])
    scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.9], [0.2, 0.8]])
    metrics = multilabel_metrics(y, scores, ["trained", "unsupported"], np.array([10, 0]))
    assert metrics["macro_auroc_over_train_supported_labels_with_both_eval_classes"] == 1.0
    assert metrics["per_label"]["unsupported"]["auroc"] == 0.0
    assert metrics["per_label"]["unsupported"]["train_positive"] == 0

# RFMiD Multi-Disease Development Run

**Status:** completed development experiment; not clinical validation or a deployable diagnostic model.

## Question and scope

This run tested whether one fine-tuned fundus-image model could learn multiple RFMiD label heads from the available public image corpus. It is not a claim to detect every retinal disease. The dataset's `Disease_Risk` target is reported separately and must not be interpreted as a universal disease diagnosis. Condition labels are source-provided annotations, and performance is specific to this dataset and its image-level split.

RFMiD's official split contains 1,920 training, 640 validation, and 640 test images (3,200 total). The local public mirror was `ctmedtech/RFMID`, revision `16ee0cf`; its dataset metadata identifies the content as CC BY 4.0. The original dataset descriptor is Pachade et al., *Retinal Fundus Multi-Disease Image Dataset (RFMiD): A Dataset for Multi-Disease Detection Research*, Data, 2021, doi: [10.3390/data6020014](https://doi.org/10.3390/data6020014). No images or raw image-level prediction rows are included in this repository release.

## Model and training

- Backbone: torchvision ConvNeXt-Tiny initialized with ImageNet weights; all model parameters were fine-tuned.
- Output: 46 independent sigmoid heads: the dataset's overall-risk target plus 45 condition labels.
- Input: RGB images resized to 320 x 320 and normalized with ImageNet channel statistics.
- Augmentation: horizontal flip, rotation up to 8 degrees, and modest brightness/contrast/saturation jitter.
- Loss: weighted multi-label binary cross-entropy. Positive weights were `sqrt(negative/positive)`, clipped to [1, 10]. A head contributed to training only when the training split contained both classes.
- Optimizer: AdamW, learning rate 2e-5, weight decay 2e-4; batch size 32; seed 20261021; mixed precision on one Google Cloud L4.
- Selection: minimum validation BCE, patience 5, minimum improvement 1e-5. The selected checkpoint was epoch 20; training stopped after epoch 25. The test partition was evaluated once after selection at a fixed 0.5 threshold.

The validation best was epoch 20, with validation macro AUROC 0.8887 and macro average precision 0.3889. Training lasted 3,566 seconds. The per-epoch history is in the aggregate run summary.

## Held-out split results

On the 640-image RFMiD test partition, the selected epoch achieved:

| Metric | Result | Interpretation |
|---|---:|---|
| Macro AUROC | 0.8939 | Mean across 36 train-supported heads with both positive and negative test examples. |
| Macro average precision | 0.4324 | Mean across 36 train-supported heads with test positives; strongly affected by prevalence and sparse labels. |
| Mean per-label accuracy, threshold 0.5 | 0.9614 | Averages binary correctness over image-label pairs; dominated by negative labels and not patient-level accuracy. |
| Mean accuracy over train-supported labels, threshold 0.5 | 0.9829 | Also prevalence-sensitive; not a summary of clinical diagnostic safety. |
| Exact multilabel subset accuracy, threshold 0.5 | 0.0344 | Fraction of images for which every label was simultaneously correct. |

The overall-risk target alone had 593/640 correct (92.66% accuracy), sensitivity 92.89% (470/506; 36 false negatives), and specificity 91.79% (123/134; 11 false positives). This does **not** meet a zero-miss objective.

For scale, nominal image-level 95% Wilson intervals are 90.37%--94.43% for overall-risk accuracy, 90.31%--94.82% for its sensitivity, and 85.90%--95.35% for its specificity. AMD sensitivity is 70.97% with a nominal interval of 53.41%--83.90%. These intervals treat images as independent because patient identifiers were unavailable; they may therefore understate uncertainty if images are correlated within patients.

| Label | Train positives | Test positives | Sensitivity | Specificity | AUROC | AP | Test FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| DR | 376 | 124 | 90.32% | 94.19% | 0.9797 | 0.9218 | 12 |
| ARMD (AMD) | 100 | 31 | 70.97% | 97.70% | 0.9660 | 0.6434 | 9 |
| MH | 317 | 104 | 89.42% | 95.90% | 0.9721 | 0.9034 | 11 |
| DN | 138 | 46 | 56.52% | 94.78% | 0.8935 | 0.4687 | 20 |
| MYA | 101 | 32 | 87.50% | 99.01% | 0.9940 | 0.9272 | 4 |
| BRVO | 73 | 23 | 73.91% | 98.70% | 0.9320 | 0.7609 | 6 |
| ODC | 282 | 91 | 51.65% | 96.36% | 0.9159 | 0.6933 | 44 |
| ODP | 65 | 24 | 20.83% | 99.03% | 0.8342 | 0.2674 | 19 |
| ODE | 58 | 17 | 82.35% | 98.72% | 0.9534 | 0.7252 | 3 |

The exact per-label values for all heads, including confusion counts and support, are provided in [`rfmid_per_label_metrics.csv`](../results/rfmid_multidisease_convnext_tiny_20261021/rfmid_per_label_metrics.csv) and [`run_summary.json`](../results/rfmid_multidisease_convnext_tiny_20261021/run_summary.json). Table values above are rounded for readability; the machine-readable files are authoritative.

## Important limitations

1. The official files provide image-level splits but no patient identifiers; patient-disjointness could not be verified. This is not an independent patient- or site-level validation.
2. Aggregate test-label support counts were inspected during the pre-run split audit. No test images or test predictions were used for model/epoch selection, but the partition should not be described as a pristine, untouched certification cohort.
3. The fixed 0.5 operating threshold was not calibrated to a clinical sensitivity, specificity, or risk target. The 36 false negatives on `Disease_Risk` and disease-specific misses show why the model cannot be represented as an autonomous clinical system.
4. Rare labels are not adequately learned or evaluated. Two condition heads had zero positive training examples and were excluded from the loss; many others had fewer than 10 training positives. A high accuracy for such a label can simply reflect its low prevalence. See support for every label in the CSV.
5. `Disease_Risk` is the dataset's own overall-risk label. It does not establish coverage of all retinal disorders, and fundus photography cannot assess conditions requiring other modalities such as OCT.
6. These are retrospective benchmark metrics on one dataset and mirror. They do not establish generalization across cameras, sites, populations, prospective workflows, or clinical benefit.

## Reproduction and artifact policy

The training entry point is [`train_rfmid_multilabel.py`](../src/active_retina/train_rfmid_multilabel.py), with focused tests in [`test_rfmid_multilabel.py`](../tests/test_rfmid_multilabel.py). The run command was:

```text
PYTHONPATH=src:pydeps OMP_NUM_THREADS=4 python3 -u -m active_retina.train_rfmid_multilabel \
  --data-root external/RFMiD_hf \
  --outdir results/rfmid_multidisease_convnext_tiny_20261021 \
  --artifact models/rfmid_convnext_tiny_multilabel_20261021.pt \
  --size 320 --epochs 30 --batch-size 32 --workers 8 \
  --max-seconds 6600 --seed 20261021
```

The trained 107 MB checkpoint was retrieved to the local ignored `models/` directory and is intentionally not committed. The public commit contains the training code, tests, aggregate metrics, and run summary; it does not redistribute the dataset, images, row-level records, or checkpoint. Artifact SHA-256: `13bf8eab863d43d83805fe81997efb354228f3233f8f93fb6c1a7b49f0d9a12d`.

Runtime on the L4: Python 3.12.3, PyTorch 2.9.1+cu129, torchvision 0.24.1+cu129, CUDA 12.9. This run used a dataset mirror, so users should independently verify the applicable dataset terms and source provenance before reuse or redistribution.

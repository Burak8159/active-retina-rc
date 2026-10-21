# Scoped Retinal Multitask Run (2026-10-21)

## Method

ConvNeXt-Tiny initialized from ImageNet weights was fine-tuned with source-masked heads for five-grade diabetic-retinopathy grading, DDR ungradable-image detection, and HYGD glaucoma GON classification. MMRDR lesion classification was skipped because its image archive was not staged. The adaptive continuation used learned homoscedastic task-loss weights, constrained to 0.5--2.0, and initialized from the best-validation fixed-weight checkpoint. This is adaptive multitask optimization, not reinforcement learning.

## Data and split

After exact-image deduplication, 9,661 images remained: 6,823 train, 1,472 validation, and 1,366 test. Six cross-split duplicate groups were found and resolved; seven same-source repeats were removed; no conflicting-label groups were found. Image-level DDR grouping is unverified. DeepDRiD, DRTiD, and HYGD patient IDs were available and used for patient-level grouping. The datasets and related project results have been used during development, so this is not independent external validation.

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| DDR | 3,002 | 625 | 608 |
| DeepDRiD | 1,124 | 253 | 212 |
| DRTiD | 2,166 | 500 | 434 |
| HYGD | 531 | 94 | 112 |

## Results

The adaptive stage selected epoch 1 by unweighted aggregate validation loss (0.422679). The fixed-weight baseline's best validation loss was 0.422679; later adaptive epochs reached 0.444722 by epoch 6, triggering early stopping. Learned active-head weights at the selected epoch were approximately DR 1.002, quality 1.002, and glaucoma 1.002 (lesion head unused). This did not establish a meaningful validation improvement over fixed weighting.

The one-time development test report for the validation-selected adaptive checkpoint:

| Task/source | N | Accuracy | Additional result |
|---|---:|---:|---|
| Referable DR, DDR | 479 | 88.94% | AUROC 0.961 |
| Referable DR, DeepDRiD | 212 | 90.09% | AUROC 0.964; patient IDs available |
| Referable DR, DRTiD | 434 | 82.95% | AUROC 0.892; patient IDs available |
| Five-grade DR, DDR | 479 | 69.10% | Macro recall 0.688 |
| Five-grade DR, DeepDRiD | 212 | 76.42% | Macro recall 0.603; patient IDs available |
| Five-grade DR, DRTiD | 434 | 65.67% | Macro recall 0.474; patient IDs available |
| Glaucoma GON, HYGD | 112 | 98.21% | Sensitivity 98.75%, specificity 96.88%; 43 patients |
| Ungradable detection, DDR | 608 | 98.19% | AUROC 0.998 |

These are distinct source/task metrics and must not be averaged or described as a single all-retinal-disease accuracy. No global disease-versus-healthy head was trained because source labels do not cover all diseases for every image. No AMD, vein-occlusion, or other disease diagnosis is supported by this model. The reported test split is development evidence, not a sealed prospective or independent clinical validation set.

## Artifacts

The run summary and epoch log are in `results/scoped_multitask_20261021_adaptive/`. Checkpoints are retained locally under `models/`; they are not included in the public repository because each exceeds common GitHub file-size limits and dataset-derived weight redistribution rights have not been verified. Row-level predictions and split manifest are also retained locally, not published.

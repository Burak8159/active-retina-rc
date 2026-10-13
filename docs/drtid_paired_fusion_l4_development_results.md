# DRTiD Paired-View L4 Development Results

## Decision

`NO_GO` for the requested 80% specificity, zero-miss, 100%-coverage binary target. The paired-view model ranked cases meaningfully above chance, but its zero-miss specificity on the development validation split remained very low. These runs are model-development evidence, not independent test or clinical certification evidence.

## Data boundary

- Source: official DRTiD training partition only; the official testing partition was excluded because it had been inspected previously.
- 1,000 paired eyes/patients were split by patient: 800 training and 200 validation, with no patient overlap.
- Validation contained 86 referable and 114 nonreferable cases.
- The validation set was used repeatedly for checkpoint/epoch inspection. Therefore all reported validation operating points are exploratory and optimistically selected; they must not be described as confirmatory.
- Manifest SHA-256: `e965cb4a7316bd6b417d5e2b04915e8e45cdfbad2eec2b10e9151f71004d9791`.
- Backbone: ImageNet-pretrained `resnet50.a1_in1k`; layer 4 and task heads were trainable, earlier encoder layers frozen.

## Results

| Candidate | AUROC-selected epoch | AUROC | Average precision | Zero-miss specificity at selected checkpoint | Highest zero-miss specificity observed across epochs* |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paired fusion, cross-entropy | 12 | 0.8562 | 0.8542 | 7.02% | 11.40% (epoch 2) |
| Paired fusion, ordinal + hard-boundary ranking | 11 | 0.8430 | 0.8317 | 6.14% | 17.54% (epoch 10) |

Both candidates had full validation coverage by construction of the zero-miss threshold. That does not make them clinically autonomous: at this threshold, virtually all negatives are referred. Neither candidate is close to the 80% specificity target.

\* The maximum across epochs uses the same repeatedly inspected validation patients and is not a valid final operating point. It is shown only to describe instability and should not be quoted as expected generalization.

## Interpretation and next decision

Adding the ordinal and hard-boundary ranking losses did not improve AUROC or the zero-miss specificity of the AUROC-selected checkpoint. Although one epoch reached 17.54% validation specificity at zero misses, choosing that epoch after looking at validation results is another layer of selection, not evidence of a robust improvement. Stop tuning this loss recipe.

The earlier combined DDR+DRTiD RTX failure audit also identified a data/target issue: DDR grade 5 denotes ungradable images in this label schema, yet the binary conversion treated grade 5 as referable disease. Several such images were severely overexposed. This supports separating image gradability/reacquisition from DR severity; it does not explain away the poor DRTiD paired-fusion specificity.

Any next experiment must materially change the intervention (for example, full-encoder fine-tuning with discriminative learning rates and a separately trained gradability gate), use only development data for iteration, and reserve a genuinely untouched patient-level cohort for one locked evaluation. No result from the runs recorded here supports forced binary diagnosis for every image.

## Follow-up experiments (2026-10-11)

Two materially different candidates were tested on the same patient-ID-separated 800/200 split of the official DRTiD training rows. The official DRTiD testing partition was not used for training or checkpoint selection.

| Candidate | Epochs retained | DRTiD holdout AUROC | Zero-miss specificity | Decision |
| --- | ---: | ---: | ---: | --- |
| Full-encoder paired fusion with DeepDRiD overall-quality auxiliary supervision | 3 (stopped early) | 0.7494 | 9.65% at epoch 2 | Reject; quality AUROC 0.531 at epoch 3 |
| Full-encoder spatial cross-attention over paired ResNet feature maps | 2 (stopped early) | 0.5829 | 4.39% at epoch 2 | Reject; substantially below the paired-fusion baseline |

The first experiment used DeepDRiD official-training quality labels only; its quality holdout AUROC remained near chance. The second is an ActiveRetina cross-field-attention experiment, **not an exact reproduction of CrossFiT**: it uses a compact three-layer token transformer, ImageNet-pretrained ResNet-50, and 224-pixel inputs, without CrossFiT's localization-derived aligned positional encoding or fundus attention mask. Both runs were stopped because their validation discrimination and zero-miss specificity were clearly below the established paired-fusion results. Neither checkpoint is promoted.

The grade-wise error audit of the selected zero-miss DRTiD source policy found 43/103 grade-0 negatives cleared and 0/24 grade-1 negatives cleared; all 123 referable eyes were referred. In the optimistic pooled per-source ceiling, non-DRTiD sources contribute 159/258 cleared negatives. Reaching 65% on the 385-negative pooled development cohort therefore requires at least 251 true-negative clearances, including at least 92/127 DRTiD negatives (72.4%) if the other source-specific counts remain unchanged. This is a development arithmetic target, not a generalization estimate.

### Next model direction

Do not continue generic pooling/attention variants or auxiliary quality transfer. The next distinct comparator is a faithful, resource-adjusted implementation of the published CrossFiT components on the DRTiD official training partition: spatial cross-field attention, shared geometry-aware positional encoding, and a localization-derived fundus attention mask. The official paper reports 88.50% macro-AUC for its full method on DRTiD, but that multiclass metric does not imply 65% specificity at zero referable misses. Any local reproduction must be labeled as such, report five-class metrics and the requested binary operating point separately, and keep the previously inspected official test partition development-only.

## Artifacts

- `results/drtid_paired_fusion_l4_dev/summary.json`
- `results/drtid_paired_fusion_l4_dev/train.log`
- `results/drtid_paired_fusion_l4_dev/validation_predictions.csv`
- `results/drtid_paired_ordinal_rank_l4_dev/summary.json`
- `results/drtid_paired_ordinal_rank_l4_dev/train.log`
- `results/drtid_paired_ordinal_rank_l4_dev/validation_predictions.csv`

Large checkpoints and image data are intentionally excluded from Git.

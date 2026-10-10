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

## Next RTX development candidate (not yet run)

The next candidate is implemented in `src/active_retina/train_drtid_paired_fusion.py` and adds a distinct per-view quality head. DRTiD supplies the paired DR task; DeepDRiD's official training partition supplies image-level `overall quality` labels (`1` means quality is adequate for retinal-disease diagnosis, `0` means it is not). The encoder is fully fine-tuned with a lower learning rate for early layers, a higher rate for layer 4, and the highest rate for task heads. The two datasets are split independently by patient; DeepDRiD official validation and online evaluation partitions are excluded.

This is a development experiment, not a guaranteed route to 80/0/100. DRTiD has no matching quality labels in the current manifest, so DeepDRiD quality performance can be measured on its patient-disjoint quality holdout, but quality-gate transport to DRTiD cannot be confirmed from DR labels alone. A quality failure can be conservatively routed to referral for a forced binary workflow, but that tradeoff must be reported in specificity and evaluated on data with independently labeled image quality.

## Artifacts

- `results/drtid_paired_fusion_l4_dev/summary.json`
- `results/drtid_paired_fusion_l4_dev/train.log`
- `results/drtid_paired_fusion_l4_dev/validation_predictions.csv`
- `results/drtid_paired_ordinal_rank_l4_dev/summary.json`
- `results/drtid_paired_ordinal_rank_l4_dev/train.log`
- `results/drtid_paired_ordinal_rank_l4_dev/validation_predictions.csv`

Large checkpoints and image data are intentionally excluded from Git.

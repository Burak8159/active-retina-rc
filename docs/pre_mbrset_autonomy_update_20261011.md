# Pre-mBRSET Autonomy Update: 2026-10-11

## Decision

`NO_GO_80_0_100_PRE_MBRSET`

The current evidence does not reach the requested 65% near-term floor, much less 80%. The best source-specialist development operating point after excluding an exact train/validation duplicate is **204/385 = 52.99% specificity**, with **0/463 observed referable misses** and a model-issued binary action for every evaluated case. The inspected validation cohort remains development-only; it has been used repeatedly for architecture, score, and threshold decisions.

## Work completed in this continuation

### Exact-duplicate integrity audit

- SHA-256 was computed over all 8,922 image rows in the three-source manifest.
- Three exact duplicate groups were found, all grade 0 and all within DDR; no cross-source duplicates or conflicting labels were found.
- One duplicate crossed train/validation. The validation copy was excluded from the clean manifest and from the corrected validation calculation. Two other groups crossed validation/test, so the validation member was retained and the test copy removed.
- A clean 8,919-row manifest and machine-readable audit are saved in `results/autonomy_route_three_dataset_manifest_20261011_exactdedup.csv` and `results/autonomy_route_three_dataset_manifest_20261011_exactdedup_summary.json`.
- This is a narrow split-integrity correction, not a new independent evaluation. The historical test set remains exposed.

### More training data: APTOS-derived public training shard

- Imported 2,487 grade-labeled training images from the public Hugging Face APTOS-derived training shard. Synthetic covariates were deliberately ignored; no validation/test data were appended.
- The training-only append utility preserves the original validation/test rows and rejects image-ID collisions or non-training additions.
- A conservative last-stage replay used source-balanced batches and retained the existing checkpoint-selection gate. Epoch 1 reduced the shared-threshold zero-miss validation specificity from 30.05% to 26.68%; epoch 0 was kept. This addition is not promoted as an improvement and does not change the source-specialist operating point.
- Dataset documentation says the images are APTOS 2019-derived and cautions users to verify the original source terms. These images do not constitute external confirmation or patient-linked paired-view evidence.

### Previously attempted distinct model pivots checked

- The retinal-pretrained EfficientNet-B5 residual/paired-only runs in the saved logs do not improve the early-grade boundary enough: the all-source residual run reached 0.56% specificity at zero misses in epoch 1 and 0% in epoch 2 on its paired validation cohort; the later paired-only retry reached 22.9% for its ordinal operating policy in epoch 1. These are not comparable cohorts to the three-source aggregate, and none is promoted.
- The RTX host is reachable and idle after these runs. No duplicate training process was started.

## Corrected operating-point accounting

The source-specialist rule was re-evaluated without the validation photo duplicated in training. DDR changed from 116/207 to 116/206 correctly cleared gradable negatives; with the selected DeepDRiD and DRTiD source experts, the aggregate is 204/385. The referable denominator stays 463 because the excluded photo is grade 0. Known-ungradable handling remained through the model quality output in the selected source policies; specificity is calculated only on grades 0-1.

| Measure | Corrected development result | Target |
|---|---:|---:|
| Specificity on gradable nonreferable cases | 52.99% (204/385) | 65% near-term; 80% main |
| Missed referable cases | 0/463 observed | 0 |
| Cases receiving a binary model action | 100% | 100% |
| Status | NO-GO; repeatedly inspected validation | No clinical or population guarantee |

## Interpretation and next model decision

The limiting error remains cross-source grade-1/grade-2 ranking, especially on DRTiD. Adding training volume from the APTOS-derived images and adapting the retinal-pretrained B5 did not materially fix it. Repeating threshold grids, score mixtures, or the same fine-tuning family on this validation set would mainly overfit the already inspected cohort. The next high-value change is genuinely new, adjudicated grade-boundary signal (or independent data that can legally be used for model development), followed by a newly untouched certification cohort. mBRSET may provide that signal, but it is not guaranteed to meet the target.

Do not claim `65/0/100`, `80/0/100`, autonomous clinical diagnosis, or certification from this work. The 100% action figure is operational output completeness only.

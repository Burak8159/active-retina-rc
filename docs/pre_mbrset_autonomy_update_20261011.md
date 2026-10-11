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

## Saved-checkpoint ceiling audit

I then evaluated 195 saved checkpoint/score-feature candidates on the same cleaned validation predictions. For each source, the checkpoint, one of four score columns, disease-risk threshold, and model-quality threshold were all selected on that source's validation rows; known-ungradable cases were routed by the model quality score. This deliberately gives the existing candidate library a generous, optimistic comparison, and is not a generalization estimate.

| Source | Best validation-only candidate | Cleared gradable negatives | Zero-miss positives |
|---|---|---:|---:|
| DDR | APTOS-replay checkpoint, max-grade margin | 118/206 (57.28%) | 0/286 |
| DeepDRiD | retinal-pretrained B5, risk score | 41/52 (78.85%) | 0/54 |
| DRTiD | source-balanced ConvNeXt, risk score | 46/127 (36.22%) | 0/123 |
| Aggregate | per-source best of 195 candidates | 205/385 (**53.25%**) | 0/463 |

Every case receives a model-issued binary action and all 133 known-ungradable DDR cases are referred in this validation audit. This is an explicitly validation-selected upper-bound diagnostic, not a frozen operating point. It remains 46 additional true-negative clearances short of 65% (at least 251/385), and 103 short of 80% (at least 308/385). The bottleneck is DRTiD: even choosing its best saved candidate leaves specificity at 36.22% with zero observed misses.

The reproducible audit is `src/active_retina/audit_validation_checkpoint_ceiling.py`; outputs are `results/validation_checkpoint_ceiling_20261011.json` and `results/validation_checkpoint_candidates_20261011.csv`. It deliberately reads only validation predictions and excludes the validation eye whose exact image is present in training. The prior exposed test partitions are not loaded.

## Interpretation and next model decision

The limiting error remains cross-source grade-1/grade-2 ranking, especially on DRTiD. Adding training volume from the APTOS-derived images and adapting the retinal-pretrained B5 did not materially fix it. Repeating threshold grids, score mixtures, or the same fine-tuning family on this validation set would mainly overfit the already inspected cohort. The next high-value change is genuinely new, adjudicated grade-boundary signal (or independent data that can legally be used for model development), followed by a newly untouched certification cohort. mBRSET may provide that signal, but it is not guaranteed to meet the target.

Do not claim `65/0/100`, `80/0/100`, autonomous clinical diagnosis, or certification from this work. The 100% action figure is operational output completeness only.

## Follow-up: IDRiD replay and patient-grouped expert audit

### IDRiD training-only addition

- The official disease-grading training partition contributed 406 unique images after excluding both copies of two conflicting duplicate-label groups and one redundant copy from each of three same-label duplicate groups. No IDRiD test image was imported or scored, and IDRiD has no patient identifiers adequate for a patient-disjoint split.
- The new RTX replay initialized from the existing ConvNeXt candidate and trained the final stage with train-only hard grade-1/grade-2 sampling. Its initial zero-miss validation point was 185/385 (**48.05%**); after epoch 1 it fell to 132/385 (**34.29%**) with 0/463 observed referable misses. The original warm-start remains the best checkpoint; epoch 1 was not promoted.

### Patient-grouped out-of-fold expert consensus

- Added `src/active_retina/audit_source_expert_consensus.py` and its unit tests. The audit uses the exact shared 981-eye cohort after excluding the known train/validation duplicate; folds keep both eyes of each patient together and thresholds/candidate ranking are fit only on the other four folds.
- With the single best fold-selected expert per source, out-of-fold specificity was 207/385 (**53.77%**) but 7/463 referable eyes were missed. Relaxing the clearance rule to reach 255/385 (**66.23%**) specificity missed 29/463 positives. The conservative union of referrals reached zero observed misses only by clearing 0/385 negatives (0% specificity).
- These results reject the tempting post-hoc ensemble interpretation: the candidate family can cross 65% only on this validation cohort when allowed to miss positives. The previous 52.99% (204/385) and 53.25% (205/385) remain same-validation development points, not independent zero-risk evidence. The patient-grouped audit is still exploratory because these checkpoint predictions and the validation cohort have informed prior project choices.
- Seven held-out misses at the single-expert setting are recorded in `results/source_expert_consensus_oof_20261011_caseaudit.json` for targeted review. They are all grade 2; they occur in DDR, DeepDRiD, and DRTiD, so the remaining failure is not isolated to one source.

### Decision

`NO_GO_65_0_100_PRE_MBRSET`. Do not deploy the permissive consensus or select its threshold on the same validation cohort. The meaningful next intervention is new grade-boundary supervision and a new independent cohort, not additional search over the current checkpoint library. The RTX is idle after the interrupted replay; its saved artifact retained the prior best checkpoint.

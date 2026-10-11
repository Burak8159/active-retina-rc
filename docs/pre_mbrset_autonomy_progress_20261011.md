# Pre-mBRSET Autonomy Progress: 2026-10-11

## Decision

`NO_GO_80_SPECIFICITY_ZERO_MISS_FULL_BINARY`

The best shared-threshold three-source candidate is **47.15% validation specificity** at zero missed referable eyes, with a binary output for every case. It clears 182/386 gradable nonreferable eyes, misses 0/463 referable eyes and 0/133 known-ungradable eyes, and emits an action for all 982 validation eyes. A source-conditioned threshold audit reaches **48.45% (187/386)** on the same validation cohort, with those same zero-miss and full-action counts. This is development evidence only: the validation set has been repeatedly inspected, source-specific thresholds were selected on that same set, the latest full-DDR fine-tune did not beat its warm start, and no test partition was loaded for that experiment. None of these results supports clinical autonomy.

The current operating objective is: zero missed gradable referable eyes (grades 2-4), a model-produced binary `CLEAR` or `REFER` for every input, and then maximize specificity among gradable nonreferable eyes (grades 0-1). Grade-5/ungradable routing is reported separately and must also come from model output. No ground-truth label may determine an action. The previously discussed 80% specificity remains a stretch target for comparison, not a reason to hide or reject a lower-specificity candidate that satisfies the two hard operating constraints on development validation.

## Experiments completed in this pass

| Candidate / diagnostic | Validation result at zero missed referable cases | Interpretation |
|---|---:|---|
| Source-conditioned thresholds for the best ConvNeXt checkpoint, one independently selected threshold per known dataset source | 187/386; **48.45%**; 0/463 missed; 0/133 known-ungradable missed; 100% binary actions | DDR 112/207, DeepDRiD 32/52, DRTiD 43/127 negatives cleared. Reproducible diagnostic in `results/source_conditioned_zero_miss_audit_20261011/summary.json`. Each threshold was fit and scored on the same repeatedly inspected validation set; source identity must be known at inference. Not independent confirmation. |
| ConvNeXt-Tiny warm-start fine-tune after adding the complete official DDR training split (5,217 new train images), ordinal-risk selection, 384px; one full epoch | Warm start: 182/386 (47.15%); trained epoch: 164/386 (**42.49%**); both 0/463 missed, 0/133 known-ungradable missed, 100% binary actions | Candidate gate retained epoch 0. Train loss 2.3187, 4,946 training batches, zero non-finite-gradient skips. Validation/test paths and labels were preserved exactly; test images were not loaded. The larger DDR addition did not improve the model. |
| ConvNeXt-Tiny 384px, all-backbone fine-tuning, 64 train-only hard grade-1/2 pairs; best epoch 1 | 178/386; **46.11%**; 0/463 missed; 0/133 known-ungradable missed; 100% binary actions | Current best on the same 982-case three-source validation set, +12 cleared eyes over its 43.01% start. Epoch 2 regressed to 44.30%. Two updates were skipped for non-finite gradients in epoch 1; best artifact is retained. Test data was not loaded. |
| Same full-backbone candidate, 256 hard-pair exposures at lower learning rate | 177/386; 45.85%; 0/463 missed; 0/133 known-ungradable missed; 100% actions | Did not beat the 46.11% warm start; one update was skipped. Promotion gate retained the warm start. |
| Cumulative ordinal loss (weight 2) plus hard grade-1/2 ranking, 64 pairs | 177/386; 45.85%; 0/463 missed; 0/133 known-ungradable missed; 100% actions | No skipped updates, but did not beat the 46.11% warm start. Retained only as an opt-in, tested training objective; not a promoted checkpoint. |
| Same 384px checkpoint evaluated at 448px without retraining | 157/386; 40.67%; 0/463 missed; 0/133 known-ungradable missed; 100% actions | Higher-resolution inference alone reduced specificity; rejected. |
| Same 384px checkpoint evaluated at 512px without retraining | 80/386; 20.73%; 0/463 missed; 0/133 known-ungradable missed; 100% actions | Larger inference-scale shift substantially degraded separation; rejected. |
| Max-logit grade margin, `max(logits[2:5]) - max(logits[0:2])`, with the same quality gate | 173/386; **44.82%**; 0/463 missed; 0/133 known-ungradable missed; 100% actions | Frozen-checkpoint, validation-only score audit; 5 fewer negative clearances than the 46.11% ordinal-probability baseline. Test split was not loaded. Rejected. |
| Grade-boundary margin, `logit(grade 2) - logit(grade 1)`, with the same quality gate | 3/386; **0.78%**; 0/463 missed; 0/133 known-ungradable missed; 100% actions | Frozen-checkpoint validation audit; this single-boundary score nearly referred every case. Rejected. Test split was not loaded. |
| DRTiD-only full-backbone fine-tuning from the current checkpoint, one epoch | 25/127; **19.69%** after epoch 1, versus 43/127 (**33.86%**) at epoch 0; 0/123 missed; 100% actions | Patient-disjoint DRTiD validation only, no test loading, no skipped gradient updates. The promotion gate retained epoch 0. Stop this source-only fine-tuning route. |
| Tie-aware ordinal-rank fusion of the best full-backbone candidate with the independent log-mean-exp candidate | Best blend: 178/386 (46.11%) at 100% weight on the current candidate; equal blend: 133/386 (34.46%); 0/463 missed and 0/133 known-ungradable missed | The second model's errors are not complementary enough to improve the operating point. Rank normalization used average ranks for ties; all weights were development diagnostics, not a deployable fitted ensemble. Rejected as a rescue route. |
| ConvNeXt-Tiny 384px, train-only hard grade-1/2 mining, ordinal-aligned rank loss; best continuation epoch 1 | 166/386; **43.01%**; 0/463 missed; 0/133 known-ungradable missed; 100% actions | Previous best before full-backbone tuning; +9 clears over its 40.67% warm start. Epoch 2 fell to 42.23%. Test data was not loaded. |
| Same hard-mined recipe, initial 64-batch training epoch | 162/386; 41.97%; 0/463 missed; 0/133 known-ungradable missed; 100% actions | Full candidate epoch; zero nonfinite-gradient skips. Hard-example scores came only from the training partition. |
| 384px last-stage lesion-auxiliary run, one mask-labeled DDR eye per batch, 64-batch diagnostic | 165/386; 42.75%; 0/463 missed; 0/133 known-ungradable missed; 100% actions | Partial-epoch diagnostic only, not a promoted candidate. |
| Same lesion-auxiliary setup, all 184 labeled eyes cycled once | 156/386; 40.41%; 0/463 missed; 0/133 known-ungradable missed; 100% actions | Worse than its 40.67% warm start; warm-start checkpoint was preserved. Lesion-presence auxiliary learning is not supported by this run. |
| Ordinal-rank loss with random source-matched grade-1/2 pairs | 56/386; 14.51%; 0/463 missed; 0/133 known-ungradable missed; 100% actions | Rejected. Pair oversampling drove excessive referrals; best-checkpoint gate retained the 40.67% warm start. |
| ConvNeXt-Tiny paired-view log-mean-exp, temperature 0.25 | 144/386 true negatives cleared; 37.31% specificity; 0/463 missed; 100% binary actions | Best reproducible single-model development candidate. The validation set selected its threshold; its test set was not loaded in this run. |
| ConvNeXt-Tiny with only the new per-view risk head trainable | 102/386 true negatives; 26.42%; zero referable/ungradable misses; 89.61% referral | Freeze-only optimization did not preserve the prior model's boundary and was stopped after epoch 1. |
| RETFound ViT-L/16 raw five-grade output, max paired-view referable probability | 0/386 true negatives cleared; 0% specificity | Raw retinal-pretrained classifier does not transfer to this three-source validation pool at a useful zero-miss binary threshold. |
| RETFound frozen embeddings plus compact binary/ordinal heads | Best: 34/386 true negatives; 8.81% specificity; 0 missed; 96.33% referral rate | Train-only fitting and validation-only thresholding. Poor grade-1 clearance and not promoted. |
| RETFound last-block adaptation, expanded class 5 for ungradable | 15/386 true negatives; 3.89% specificity; 0/463 missed referable and 0/133 missed ungradable; 98.47% referral rate | Stopped after epoch 1 by the prespecified gate because it was far below the 37.31% baseline. All earlier transformer blocks stayed frozen. |
| Paired-field-only contribution audit, DRTiD + DeepDRiD | Macula-only: 18/179 (10.06%); optic-disc-only: 1/179 (0.56%) | No hidden high-performing individual field. These are validation diagnostics, not standalone field certifications. |
| Log-mean-exp pooling sensitivity | Temperature 0.25/0.50: 37.31%; max: 36.53%; mean: 27.46% | Pooling changes cannot close the gap. |

The prior base manifest contains 6,348 training eyes and 982 validation eyes across DDR, DRTiD, and DeepDRiD. The expanded manifest adds 5,217 nonduplicate official DDR training images; 1,618 image paths were already present and skipped. It contains 11,565 training rows, with the original 1,338 validation and 1,236 test rows preserved verbatim (held-out SHA-256 `72ae8fd2c6fdd9fbf4b5b1aca55321c9dd87b60b9997a8400fd5b84c46008902`). The expanded run did not load test images. Feature-fitting and threshold selection are development steps, not external confirmation.

One initial full-backbone run accidentally used the older two-source manifest (DDR + DRTiD; 876 validation eyes). It reached 47.60% on that different cohort but had 11 skipped gradient updates, so it is not compared with or substituted for the three-source result. Subsequent runs used the hash-verified three-source manifest.

## Boundary diagnosis and current gap

- The shared-threshold 47.15% candidate clears 169/274 grade-0 eyes (61.68%) and 13/112 grade-1 eyes (11.61%). Grade-1 clearance remains the dominant specificity bottleneck.
- The shared threshold clears 182/386 negatives; reaching 80% on this validation set requires at least 309/386, a gap of 127. The source-conditioned diagnostic clears 187, leaving a gap of 122. Even if every grade-0 eye cleared, at least 35 grade-1 eyes would have to clear; the shared-threshold candidate clears 13.
- Per-source specificity at the selected global zero-miss policy is 53.14% on DDR, 50.00% on DeepDRiD, and 33.07% on DRTiD. The aggregate remains domain-sensitive.
- DDR lesion-mask labels were joined only to eligible official training images. The join excluded annotation images colliding with held-out manifest partitions; mask-derived labels were never added to validation or test. The lesion auxiliary objective did not improve the full pass.
- Pairwise ranking now supports the deployed ordinal referable score directly. Hard-example mining scores grade-1/2 eyes from the training split only, within dataset source; it does not mine validation or test examples. This produced a modest complete-candidate gain, not a route to the 80% target by itself.
- All-backbone fine-tuning improved the zero-miss point on this development validation set, but the second epoch and a longer hard-pair run both regressed. Source-specific calibration on the current checkpoint reaches 187/386 (48.45%) on the same validation set. This is explicitly a same-set diagnostic; do not interpret it as an independent model gain or transfer its thresholds to an unidentified future source.
- Expanding training with every official DDR training image added 5,217 previously absent eyes, mostly grade 0 (2,833 additions) and grade 2 (1,938 additions), but only 15 grade-1 examples. One full 384px epoch regressed from 47.15% to 42.49%; the warm-start gate preserved the existing weights. More of the same single-image DDR data did not resolve the grade-1 boundary.
- A tie-aware rank blend of the best full-backbone model and the separate log-mean-exp model did not help: equal weighting reduced zero-miss specificity from 46.11% to 34.46%, and performance declined monotonically as weight moved away from the current candidate. The simple ensemble route is exhausted for these checkpoints.
- A max-logit class margin was also tested on the frozen checkpoint with no training and no test loading. It cleared 173/386 negatives (44.82%) at zero observed referable/ungradable misses, five fewer than the 178/386 ordinal-probability point. It is not promoted.
- A direct grade-2-minus-grade-1 logit score collapsed to 3/386 cleared negatives (0.78%) at zero observed misses. The two-class boundary alone loses useful grade-0 evidence and is rejected.
- Fine-tuning the current model on DRTiD training data alone reduced DRTiD validation specificity from 33.86% at the frozen starting point to 19.69% after one epoch, with zero observed misses and no numerical skips. The automatic best-checkpoint gate kept the starting weights; do not continue this source-only adaptation family.
- The ordinal disease head has five logits for grades 0-4; ungradable detection is a separate quality head. Thus, removing an assumed sixth ungradable disease logit would not define a distinct score and must not be treated as an unexplored route.
- The lowest-scored positive on the 46.11% candidate is a DDR grade-2 eye (`007-0693-000`, ordinal risk 0.00320); DRTiD eye `0905` is the next-lowest positive (0.00345). Neither was moved into training or relabeled. The earlier visual inspection of `0905` remains a diagnostic only, not evidence of annotation error.

## Closest numeric result is not the strongest evidence

A validation-selected score mixture reached 44.13% specificity with zero validation misses. It was chosen from 4,026 combinations. Its already-exposed diagnostic test readout was 33.13% specificity with one miss among 946 referable cases. The test result is not confirmatory and the large candidate search makes the validation maximum optimistic. Do not promote this mixture as meeting the zero-miss goal.

The previous 37.31% ConvNeXt point remains in the table above for historical context. The current 46.11% single-model point still needs 131 additional negative clearances to reach 80% on validation. The remaining gap is a ranking/separation problem, especially grade 1, not a threshold typo.

## Corrections and reproducibility

- Added a validation-only audit of per-view risk logits and pooling modes.
- Added a direct RETFound checkpoint load that strictly verifies all tensors against ViT-L/16, 256-pixel input, and average pooling. The public model's older Hub metadata could not be consumed through the installed `timm` Hub adapter, so the cached state dictionary was loaded explicitly and strictly.
- RETFound weights: [official model repository](https://huggingface.co/bswift/RETfound_eyepacs_DR); loaded safetensors SHA-256 `9b7d5a3395d6b7839278327afce723f388398b7a0c5d24f864ecaa6f39683bd0` (1,213,500,972 bytes).
- Added a frozen-embedding training comparison and a last-block fine-tuning path with six-way outputs (grades 0-4 plus ungradable), cumulative ordinal loss, and an epoch-1 stopping gate.
- Added and unit-tested an optional cumulative ordinal training loss for the mixed-view model; the weight-2 RTX trial did not improve the promoted validation checkpoint.
- Fixed the new audit's action rule: grade-5 status no longer forces referral using the answer label. The report measures grade-5 referral rate from the model's own score.
- No test images were used in the RETFound experiments. The full-binary mixture's test numbers are retained only as exposed diagnostics.

Machine-readable evidence:

- `results/mil_view_contribution_audit_20261011.json`
- `results/retfound_embedding_autonomy_20261011.json`
- `results/retfound_lastblock_gate_20261011_summary.json`
- `results/augmented_full_binary_gap_validation_audit_20261011_clean/summary.json`
- `results/max_grade_margin_audit_20261011/summary.json` and `validation_predictions.csv`
- `results/grade_boundary_margin_audit_20261011/summary.json` and `validation_predictions.csv`
- `results/rtx_drtid_domain_adapt_20261011/summary.json` and `validation_predictions.csv`
- `results/ddr_full_train_manifest_20261011_summary.json` and `results/autonomy_route_full_ddr_train_20261011_summary.json`
- `results/rtx_convnext_full_ddr_ordinal_20261011/summary.json` and `validation_predictions.csv`
- `results/source_conditioned_zero_miss_audit_20261011/summary.json` and `predictions.csv`

## Next decision

Preserve the source-balanced ConvNeXt as the best shared-threshold development candidate; it achieved 47.15% (182/386) at zero observed misses on the inspected three-source validation set. A known-source threshold diagnostic reaches 48.45% (187/386), but uses the same validation cohort for calibration and scoring. The full-DDR expansion regressed and was not promoted. The 80% target remains 127 clearances away under a shared threshold, or 122 under the same-set source-conditioned diagnostic. Tested sampler, pooling, resolution, loss, score, and source-only fine-tuning pivots have not closed the grade-1/grade-2 ranking gap. The next substantial gain requires a materially different pretrained representation or new independently adjudicated grade-boundary data; mBRSET may supply useful signal, but it cannot be assumed to solve the gap. Any autonomy claim still requires a frozen candidate evaluated on a genuinely untouched patient-level cohort.

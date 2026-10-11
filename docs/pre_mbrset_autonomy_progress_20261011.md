# Pre-mBRSET Autonomy Progress: 2026-10-11

## Decision

`NO_GO_80_SPECIFICITY_ZERO_MISS_FULL_BINARY`

The best complete current candidate is **43.01% validation specificity** at zero missed referable eyes, with a binary output for every case. It clears 166/386 gradable nonreferable eyes, misses 0/463 referable eyes and 0/133 known-ungradable eyes, and emits an action for all 982 validation eyes. This is development evidence only: the validation set has been repeatedly inspected, no test partition was loaded in these runs, and none of these results supports clinical autonomy.

The target is 80% specificity among gradable nonreferable eyes (grades 0-1), zero missed gradable referable eyes (grades 2-4), and a model-produced binary `CLEAR` or `REFER` for every input. Grade-5/ungradable routing is reported separately and must also come from model output. No ground-truth label may determine an action.

## Experiments completed in this pass

| Candidate / diagnostic | Validation result at zero missed referable cases | Interpretation |
|---|---:|---|
| ConvNeXt-Tiny 384px, train-only hard grade-1/2 mining, ordinal-aligned rank loss; best continuation epoch 1 | 166/386 true negatives; **43.01%** specificity; 0/463 missed; 0/133 known-ungradable missed; 100% binary actions | Best complete candidate so far. It improves on the 40.67% warm-start by 9 cleared eyes. Epoch 2 fell to 42.23%, so epoch 1 was retained. Test data was not loaded. |
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

The training/validation manifest contains 4,675 training eyes and 982 validation eyes across DDR, DRTiD, and DeepDRiD. The RETFound runs loaded zero test images and verified no patient overlap between train and validation. Feature-fitting and threshold selection are development steps, not external confirmation.

## Boundary diagnosis and current gap

- The 43.01% candidate clears 159/274 grade-0 eyes (58.03%) but only 7/112 grade-1 eyes (6.25%). Grade-1 clearance, not another global threshold adjustment, is the dominant specificity bottleneck.
- The threshold requires 166 clearances today; reaching 80% on this validation set requires at least 309/386, a gap of 143. Even if every grade-0 eye cleared, at least 35 grade-1 eyes would have to clear; only 7 currently do.
- Per-source specificity at the selected zero-miss policy is 47.83% on DDR, 50.00% on DeepDRiD, and 32.28% on DRTiD. The aggregate remains domain-sensitive.
- DDR lesion-mask labels were joined only to eligible official training images. The join excluded annotation images colliding with held-out manifest partitions; mask-derived labels were never added to validation or test. The lesion auxiliary objective did not improve the full pass.
- Pairwise ranking now supports the deployed ordinal referable score directly. Hard-example mining scores grade-1/2 eyes from the training split only, within dataset source; it does not mine validation or test examples. This produced a modest complete-candidate gain, not a route to the 80% target by itself.
- The DRTiD validation grade-2 eye `0905` remains the lowest-scored positive on the 43.01% candidate (ordinal risk 0.00128) and looked visually subtle in both paired views. It was not moved into training or relabeled; this observation is a development diagnostic, not evidence of annotation error.

## Closest numeric result is not the strongest evidence

A validation-selected score mixture reached 44.13% specificity with zero validation misses. It was chosen from 4,026 combinations. Its already-exposed diagnostic test readout was 33.13% specificity with one miss among 946 referable cases. The test result is not confirmatory and the large candidate search makes the validation maximum optimistic. Do not promote this mixture as meeting the zero-miss goal.

For the current single-model ConvNeXt point, reaching 80% on the 386 validation negatives would require clearing at least 309; it clears 144. The remaining gap is a ranking/separation problem, especially grade 1, not a threshold typo. Its grade-1 specificity was 13.39% (15/112). Across sources, the current zero-miss point ranged from 18.11% on DRTiD and 25.00% on DeepDRiD to 52.17% on DDR.

## Corrections and reproducibility

- Added a validation-only audit of per-view risk logits and pooling modes.
- Added a direct RETFound checkpoint load that strictly verifies all tensors against ViT-L/16, 256-pixel input, and average pooling. The public model's older Hub metadata could not be consumed through the installed `timm` Hub adapter, so the cached state dictionary was loaded explicitly and strictly.
- RETFound weights: [official model repository](https://huggingface.co/bswift/RETfound_eyepacs_DR); loaded safetensors SHA-256 `9b7d5a3395d6b7839278327afce723f388398b7a0c5d24f864ecaa6f39683bd0` (1,213,500,972 bytes).
- Added a frozen-embedding training comparison and a last-block fine-tuning path with six-way outputs (grades 0-4 plus ungradable), cumulative ordinal loss, and an epoch-1 stopping gate.
- Fixed the new audit's action rule: grade-5 status no longer forces referral using the answer label. The report measures grade-5 referral rate from the model's own score.
- No test images were used in the RETFound experiments. The full-binary mixture's test numbers are retained only as exposed diagnostics.

Machine-readable evidence:

- `results/mil_view_contribution_audit_20261011.json`
- `results/retfound_embedding_autonomy_20261011.json`
- `results/retfound_lastblock_gate_20261011_summary.json`
- `results/augmented_full_binary_gap_validation_audit_20261011_clean/summary.json`

## Next decision

Preserve the hard-mined ConvNeXt continuation epoch-1 artifact as the current single-model development candidate; do not promote its diagnostic checkpoints or the worse later epoch. The objective/sampler experiments above establish only a modest improvement, while the 80% target remains far away. The next substantial gain likely requires more representative grade-1/grade-2 evidence or independently adjudicated label review; mBRSET may supply useful signal, but it cannot be assumed to solve the gap. Any autonomy claim still requires a frozen candidate evaluated on a genuinely untouched patient-level cohort.

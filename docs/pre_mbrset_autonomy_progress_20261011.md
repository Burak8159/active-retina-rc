# Pre-mBRSET Autonomy Progress: 2026-10-11

## Decision

`NO_GO_80_SPECIFICITY_ZERO_MISS_FULL_BINARY`

The best defensible current model result is 37.31% validation specificity at zero missed referable eyes, with a binary output for every case. This is development evidence only. No available partition is an untouched certification cohort, and none of these results supports clinical autonomy.

The target is 80% specificity among gradable nonreferable eyes (grades 0-1), zero missed gradable referable eyes (grades 2-4), and a model-produced binary `CLEAR` or `REFER` for every input. Grade-5/ungradable routing is reported separately and must also come from model output. No ground-truth label may determine an action.

## Experiments completed in this pass

| Candidate / diagnostic | Validation result at zero missed referable cases | Interpretation |
|---|---:|---|
| ConvNeXt-Tiny paired-view log-mean-exp, temperature 0.25 | 144/386 true negatives cleared; 37.31% specificity; 0/463 missed; 100% binary actions | Best reproducible single-model development candidate. The validation set selected its threshold; its test set was not loaded in this run. |
| ConvNeXt-Tiny with only the new per-view risk head trainable | 102/386 true negatives; 26.42%; zero referable/ungradable misses; 89.61% referral | Freeze-only optimization did not preserve the prior model's boundary and was stopped after epoch 1. |
| RETFound ViT-L/16 raw five-grade output, max paired-view referable probability | 0/386 true negatives cleared; 0% specificity | Raw retinal-pretrained classifier does not transfer to this three-source validation pool at a useful zero-miss binary threshold. |
| RETFound frozen embeddings plus compact binary/ordinal heads | Best: 34/386 true negatives; 8.81% specificity; 0 missed; 96.33% referral rate | Train-only fitting and validation-only thresholding. Poor grade-1 clearance and not promoted. |
| RETFound last-block adaptation, expanded class 5 for ungradable | 15/386 true negatives; 3.89% specificity; 0/463 missed referable and 0/133 missed ungradable; 98.47% referral rate | Stopped after epoch 1 by the prespecified gate because it was far below the 37.31% baseline. All earlier transformer blocks stayed frozen. |
| Paired-field-only contribution audit, DRTiD + DeepDRiD | Macula-only: 18/179 (10.06%); optic-disc-only: 1/179 (0.56%) | No hidden high-performing individual field. These are validation diagnostics, not standalone field certifications. |
| Log-mean-exp pooling sensitivity | Temperature 0.25/0.50: 37.31%; max: 36.53%; mean: 27.46% | Pooling changes cannot close the gap. |

The training/validation manifest contains 4,675 training eyes and 982 validation eyes across DDR, DRTiD, and DeepDRiD. The RETFound runs loaded zero test images and verified no patient overlap between train and validation. Feature-fitting and threshold selection are development steps, not external confirmation.

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

Do not extend the RETFound heads or the stopped last-block run. Do not run more threshold, pooling-temperature, or sampler variants on the same validation evidence. Preserve the ConvNeXt epoch-1 artifact as the current single-model development baseline. The next model experiment needs a genuinely new source of labeled boundary information or an independently justified image/label correction; mBRSET may supply that signal, but it cannot be assumed to solve the gap. Any eventual autonomy claim still requires a frozen candidate evaluated on a genuinely untouched patient-level cohort.

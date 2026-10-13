# MMRDR Transfer and RETFound Development Results

**Status:** exploratory development only; official MMRDR test partitions were not evaluated.

## Data provenance and cohort handling

The official MMRDR descriptor states that CFP images come from OIA-DDR, while UWF images come from a separate Qingdao Eye Hospital cohort. The subsets number their `tr*`/`ts*` files independently; same-numbered CFP and UWF filenames are not paired-eye identifiers. The CFP split is image-level because the source lacks patient identifiers, while the UWF official split is patient-level. This project uses an image-level train/validation carve-out from UWF's official training subset because exported patient IDs are unavailable. Consequently, these development validation results are not patient-independent confirmation.

The temporary inner archive was reconstructed on the RTX from nine size-verified pieces (18,610,317,731 bytes), used to extract only the 8,893 CFP official `tr*` images and their filtered training annotations, then removed. No CFP `ts*` images/labels were extracted for this experiment. UWF official test rows were not included in the development manifest or accessed by model code.

An initial filename join was audited and discarded as invalid. Its high referable-label disagreement confirmed that matching numeric suffixes cannot establish correspondence. The corrected manifest treats CFP as an independent auxiliary training cohort and does not pair it with UWF.

## Experiments

| Experiment | Validation result at zero observed misses | Interpretation |
| --- | ---: | --- |
| Frozen ResNet-50 UWF-only HGB, 224 px | 47/600 true negatives; 7.83% specificity; AUROC 0.900 | Same-encoder control |
| Frozen ResNet-50 CFP+UWF HGB, 224 px | 18/600 true negatives; 3.00%; AUROC 0.891 | Naive source pooling harmed UWF validation performance |
| RETFound-EyePACS direct probabilities, UWF | 26/600; 4.33%; AUROC 0.777 | Material domain shift from its EyePACS color-fundus task to MMRDR-UWF |
| RETFound features + HGB | 1/600; 0.17%; AUROC 0.888 | No useful gain |
| RETFound features + RBF-SVM | 60/600; 10.00%; AUROC 0.916 | Best single RETFound head |
| RETFound features + logistic head | 63/600; 10.50%; AUROC 0.919 | Best single RETFound head by this zero-miss criterion |
| Existing frozen ResNet+DINO HGB rank + 448-px fine-tuned ResNet rank | 202/600; 33.67%; AUROC 0.932 | Existing internal validation reference |
| Reference + 25% RETFound logistic rank | 207/600; 34.50%; AUROC 0.940 | Small same-validation gain only; five extra negatives cleared |

All rows use 572 referable and 600 nonreferable UWF validation images, with referable defined as grade >= 2. The threshold is the minimum positive validation score, so zero misses are guaranteed on these same validation observations by construction. Model choice, fusion weights, and threshold selection reused this validation set; the figures are optimistic diagnostics, not a certificate or an unbiased estimate of future performance.

### MMRDR-initialized all-source fine-tuning diagnostic

An RTX run initialized an ImageNet-pretrained ResNet-50 from the MMRDR-UWF checkpoint and fine-tuned all 23,520,326 parameters on the available DDR, DeepDRiD, DRTiD, and IDRiD training data. The run's separate validation manifest contained 1,485 gradable records (820 nonreferable and 665 referable) plus 95 ungradable records. The best selected epoch was 8: at the validation-derived zero-miss operating point it cleared 140/820 nonreferable records (17.07% specificity), referred all 665 referable and all 95 ungradable records, and assigned an action to every supplied row. The complete 0--8 epoch history is preserved in the aggregate summary.

This is a development diagnostic, not external or untouched certification. The epoch and operating threshold were selected using this validation cohort; no independent test was evaluated (`test_evaluated: false`). The result does not establish generalization to a new site, zero population risk, or clinical autonomy. It also does not replace the 70.13% same-cohort score-policy replay: it is a distinct raw-image transfer-training experiment with a different artifact and validation procedure. The low specificity indicates that this training path did not solve the operating-point objective, despite zero observed validation misses.

## Reproducible artifacts

- `src/active_retina/build_mmrdr_transfer_manifest.py` builds an explicitly non-paired CFP-training/UWF-development manifest and refuses non-training CFP rows.
- `src/active_retina/cache_mmrdr_features.py` now records source-dataset provenance in feature caches.
- `src/active_retina/cache_mmrdr_retfound_features.py` loads the public RETFound-EyePACS ViT-Large checkpoint strictly and extracts development-only features/logits.
- `src/active_retina/analyze_mmrdr_embedding_heads.py` supports source-filtered train/validation comparison and records source counts.
- `src/active_retina/analyze_mmrdr_retfound_fusion.py` performs the bounded validation-only fusion check.
- `results/mmrdr_cfp_to_uwf_transfer_20261013.json`
- `results/mmrdr_uwf_only_resnet224_control_20261013/summary.json`
- `results/mmrdr_cfp_to_uwf_resnet224_transfer_20261013/summary.json`
- `results/mmrdr_uwf_retfound_eyepacs_heads_20261013/summary.json`
- `results/clean_original_train_mmrdr_finetune_20261013/summary.json` (RTX-retrieved aggregate; patient-level split manifest and per-record predictions are retained locally but are not published here)
- `results/mmrdr_retfound_fusion_20261013/summary.json`

The RETFound checkpoint documents an EyePACS-specific five-class task and research/education use. Its matching class order is 0-No DR, 1-Mild NPDR, 2-Moderate NPDR, 3-Severe NPDR, 4-PDR; strict loading into `vit_large_patch16_224` succeeded. It is a candidate feature extractor, not a validated clinical model.

## Decision

The fixed internal UWF validation result is now 34.50% zero-miss specificity for one exploratory fused candidate, still far below the aspirational 75% target. The all-source MMRDR-initialized fine-tune reached 17.07% at zero observed validation misses, also without an independent test. Do not promote either as a deployed threshold or as clinical/autonomous validation. The main actionable learning is that CFP/UWF cannot be paired by filename, naive pooled-feature training hurts, and the tested transfer/training paths have not produced a strong zero-miss operating point. Keep any official test split sealed until model selection is frozen.

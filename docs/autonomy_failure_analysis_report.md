# ActiveRetina Autonomy Failure Analysis

Status: **analysis complete; do not rerun failed backbone loops**.

## 1. False-Negative Overlap

Analyzed 6 DDR model families on 1435 common test images with 946 referable positives. At validation-selected 99% sensitivity, **0 referable images were missed by every model**. Mean non-duplicate pairwise false-negative Jaccard was **0.120**. This means the failures are mostly model-specific and fusion/hard-example learning is justified. See `results/autonomy_false_negative_overlap_ddr.csv`, `results/autonomy_false_negative_jaccard_ddr.csv`, and `results/autonomy_persistent_false_negative_cases_ddr.csv`.

## 2. Operating Frontier

Best current forced-binary DDR 99% sensitivity operating point is `mlp_ensemble`: test sensitivity **98.6%**, test specificity **38.9%**, false negatives **13**. This fails the 90%+/zero-miss autonomy target. The frontier plot is `plots/autonomy_frontier_ddr.svg`.

## 3. Selective Screening

Best selective policy is `mlp_ensemble`: test coverage **68.8%**, test sensitivity **98.8%**, test auto-specificity **82.6%**, false negatives **11**. Selective screening is closer, but still not the final autonomous result.

## 4. Multi-View Rescue

DeepDRiD paired-view rescue is the strongest positive signal: suffix1_initial rescued 3/4 first-view false negatives (75.0%); suffix2_initial rescued 4/5 first-view false negatives (80.0%). This directly supports ActiveRetina's central claim that acquiring another field can fix errors that a first-view-only autonomy model would otherwise make.

## 5. Decision

**NO-GO for the current 90%+/zero-miss autonomous model; GO for targeted ActiveRetina-Autonomy-v2 training.** The concrete solution is not another isolated threshold run. It is an active selective fusion system: first-view risk, second-view acquisition when unsafe, model/view fusion, then CLEAR / REFER / ABSTAIN with risk-controlled thresholds. Because the false negatives are complementary across models and mostly rescued by an added view, this is a scientifically justified next model rather than blind retrying.

## 6. Paper Impact

This strengthens the flagship paper if we frame it correctly: the paper is not a leaderboard classifier. It is a falsifiable active-acquisition/autonomous-clearance study showing why single-image thresholding fails, how second-view acquisition rescues false negatives, and what evidence is required before autonomous clearance can be claimed.

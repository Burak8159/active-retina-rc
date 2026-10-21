# Patient-Grouped Expert-Selector Experiments (2026-10-12)

## Purpose

Test whether expert-score complementarity can be converted into a selector
without reusing a case's label to set its own threshold. The goal was to improve
the existing 70.13% retrospective pooled reference toward 75% specificity,
zero observed misses, and 100% binary assignment.

## Evidence Boundary

All cohorts and model outputs have already been inspected during development.
The score-channel shortlist was also derived from the earlier full-cohort,
label-informed oracle audit. Thus patient-grouped folds prevent a patient's
rows from crossing selector fit/evaluation within this experiment, but this is
still exposed exploratory development, not independent validation, risk
certification, or clinical evidence. Thresholds are fold-calibrated; zero
observed misses in this report are not a guarantee.

## Results

| Policy, outer patient-grouped evaluation | TN / 385 negatives | Specificity | Misses | Binary coverage |
|---|---:|---:|---:|---:|
| Historical source-specific score-fusion reference, full-cohort post-hoc thresholds | 270 / 385 | 70.13% | 0 | 100% |
| Patient-fold pairwise rank selector, train-fold minimum-positive thresholds | 245 / 385 | 63.64% | 5 | 100% |
| Nested regularized blend with cross-fitted positive-score thresholds | 216 / 385 | 56.10% | 1 | 100% |
| Nested single-expert route, selected in inner folds | 225 / 385 | 58.44% | 2 | 100% |
| Nested OR across complementary experts, shrink selected in inner folds | 206 / 385 | 53.51% | 1 | 100% |
| Nested nonlinear selector (logistic, pairwise rank, shallow tree) | 196 / 385 | 50.91% | 1 | 100% |
| Earlier full-cohort label-informed oracle union (not a model) | 293 / 385 | 76.10% | 0 | 100% |

The table's retained pooled aggregates are in `results/expert_selector_nested_crossfit_20261012.json` (pairwise rank), `results/expert_selector_nested_crossfit_oof_threshold_20261012.json` (regularized blend, single-expert route, and OR route), `results/expert_selector_nested_crossfit_nonlinear_20261012.json` (nonlinear selector), and `results/expert_score_oracle_union_20261012.json` (label-informed oracle). The historical replay row comes from `results/deployable_exposed_baseline_7013_20261013_with_components/summary.json`. The single-expert and OR pooled counts are sums of the corresponding per-source aggregates in the cross-fitted threshold summary. Rows without a retained source for their exact aggregate are intentionally not included in the manuscript comparison.

## Final Simple-Blend Diagnostic

After the nested selectors failed to recover the oracle complementarity, I
tested every stored per-source score as a single convex blend with the current
source-specific recipe. The grid used 101 weights from 0 to 1, with the
zero-miss cutoff re-derived from the same evaluated cohort for each candidate.

| Source | Current TN | Best one-score blend TN | Score channel | Blend weight on channel |
|---|---:|---:|---|---:|
| DDR | 134 / 206 | 135 / 206 | `grade_boundary_margin_audit_20261011::ordinal_risk` | 0.12 |
| DeepDRiD | 51 / 52 | 51 / 52 | Current recipe | 0.00 |
| DRTiD | 85 / 127 | 86 / 127 | `rtx_drtid_rankheavy_viewaux_current_split/pooled_validation_hflip_predictions.csv::paired` | 0.10 |
| **Pooled** | **270 / 385** | **272 / 385** | | |

The apparent 0.52-point lift (two eyes) is post-hoc and uses labels both to
choose the blend and to set its cutoff. It is not a newly trained model,
patient-held-out gain, or deployable policy, and it is rejected as progress
toward the 75% target. It does show that this one-channel convex-blend family
does not unlock the available oracle margin. Do not repeat this grid.

The rows are not equally calibrated: the 70.13% historical reference and 76.10%
oracle use full-cohort labels to choose thresholds, whereas the new selector
uses patient-grouped cross-fitting. Therefore, the cross-fit rows are not
evidence that a deployable model is below the reference by exactly the listed
margin; they show that the tested learning/routing rules do not recover the
oracle complementarity. The oracle remains explicitly label-informed and
non-deployable.

The source-wise reference counts are DDR 134/206, DeepDRiD 51/52, and DRTiD
85/127. The latest RTX fine-tune's paired DRTiD score cleared 68/127 at its
post-hoc zero-miss threshold, below the incumbent 85/127; no training process
was running at inspection time.

## Data Accounting

The comparison contains 981 eye cases from 928 unique patients: DDR 625/625,
DeepDRiD 106/53, and DRTiD 250/250. All outer and inner splits use
`StratifiedGroupKFold` with `patient_id`; both DeepDRiD eyes remain together.
Specificity is computed over the 385 gradable nonreferable eyes. Forced
referrals stay assigned a binary action and count as false referrals when
their labels are nonreferable.

## Decision

The score-only expert-routing family failed the 75/0/100 development target and
did not improve on the 70.13% historical reference. Do not continue tuning
these same blend, single-score, OR, or shallow-tree variants. The next
materially different route must create new image-level discrimination using
training patients only, then be evaluated on a held-out patient cohort. The
current evidence does not establish that route will reach 75%.

## Reproduction

```sh
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.evaluate_crossfit_expert_selector \
  --output results/expert_selector_nested_crossfit_nonlinear_20261012.json
```

The saved JSON contains per-source folds, inner model selections, threshold
calibration values, and the learned-blend, single-expert, and OR-policy results.

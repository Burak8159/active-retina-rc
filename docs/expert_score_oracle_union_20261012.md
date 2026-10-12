# Expert-Score Union Audit (2026-10-12)

## Result

An exhaustive scan of stored prediction-score columns found a **label-informed
oracle union of 293/385 = 76.10% specificity**, with zero observed required-case
misses and binary assignments for all evaluated eyes. This exceeds the 75%
count threshold of 289/385 by four true negatives and is 23 cases above the
historical 270/385 (70.13%) development reference.

| Source | Current recipe TN | Oracle-union TN / negatives | Specificity |
|---|---:|---:|---:|
| DDR | 134 | 148 / 206 | 71.84% |
| DeepDRiD | 51 | 51 / 52 | 98.08% |
| DRTiD | 85 | 94 / 127 | 74.02% |
| **Pooled** | **270** | **293 / 385** | **76.10%** |

## What This Means

This is an **upper bound over candidate scores, not an achieved model or
deployable policy**. For each score column, the audit sets its zero-miss cutoff
to the minimum score of the required-positive/ungradable cases in the same
evaluated cohort, then clears a negative if any candidate score falls below its
own cutoff. Thus both the per-expert cutoffs and the final union depend on the
evaluation labels. The audit is intentionally post-hoc, exposed, and unsuitable
for clinical use or a publication claim of 75% generalization.

The meaningful engineering finding is narrower: the stored model outputs do
contain enough *complementary retrospective signal* to exceed 75% if an
independent, label-free expert selector can recover at least 19 of the 23
additional negatives without missing positives. The current candidate model
blends and patient-grouped cross-fit stacks have not done that. The selected
patient-grouped logistic-stack attempts remain below 75 and must not be
substituted for the oracle number.

## Reproduction

From the project root:

```sh
PYTHONPATH=src python -m active_retina.audit_expert_score_oracle_union
```

The JSON output records source counts and SHA-256 hashes of prediction files
used. The number of score columns may change when more outputs are added; this
does not change the label-informed status of the calculation.

## Next Valid Gate

To claim a genuine 75% model result, train or calibrate the expert selector
using patient-disjoint development predictions, freeze its model and
thresholds, and evaluate it on patients not used for selection. The RTX host
must be reachable for another image-level DRTiD training run; its SSH port was
closed during this audit. Keep the 70.13% record unchanged as a historical
comparator. Any candidate above it gets a new result ID and provenance record.

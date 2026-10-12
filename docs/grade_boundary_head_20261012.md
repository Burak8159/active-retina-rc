# Grade-Boundary Head Experiment (2026-10-12)

## Decision

The frozen-feature grade-boundary head did not improve the current pooled
development result. Superseding the earlier freeze direction: 70.13% is a
historical comparison baseline only, not the frozen final model. Development
remains open. Do not spend more runs on this head or its loss-weight variants.

## Experiment

`src/active_retina/train_grade_boundary_head.py` trains source-specific MLP
heads over cached 2048-dimensional ResNet-50 features. Paired eyes use the
view-order-invariant concatenation of the per-eye mean and absolute view
difference; single-view DDR records use a zero difference vector. The objective
combines weighted binary referable-DR loss and a pairwise ranking loss. The
grade-1 negative class receives increased weight.

Training used only the datasets' named training rows. Exact comparison eyes
overlapping those training rows were removed before fitting (DDR: 251; DeepDRiD:
92; DRTiD: 156). The score comparison uses the same exposed rows as the current
70.13% audit: DDR 625 eyes, DeepDRiD 106 eyes, and DRTiD 250 eyes. DRTiD's
previously inspected test data remain development-only. Each candidate uses a
post-hoc minimum-positive-score threshold, so the reported zero misses are not
a risk guarantee and are not independent evaluation.

## Result

| Source | Best TN / gradable negatives | Specificity | Misses | Binary coverage |
|---|---:|---:|---:|---:|
| DDR | 23 / 206 | 11.17% | 0 | 100% |
| DeepDRiD | 27 / 52 | 51.92% | 0 | 100% |
| DRTiD | 11 / 127 | 8.66% | 0 | 100% |
| Pooled, best source head per source | 61 / 385 | 15.84% | 0 | 100% |

By comparison, the existing score-fusion reference is 270 / 385 = 70.13% on
these same gradable negatives, with zero observed misses and binary action for
all 981 comparison eyes. The new head is 209 true negatives behind, so it is
not a candidate for replacement. Blending each trained score with its existing
source score over 101 convex weights recovered zero additional true negatives:
the pooled result remained 270 / 385.

## What the Failure Says

The frozen ResNet feature representation plus this binary/ranking objective did
not order the difficult negatives below the least-risk positive. More epochs or
minor loss-weight changes are not justified by these results. The next material
experiment needs new image-level signal (full-image fine-tuning or a genuinely
different pretrained retinal representation) and must preserve the source and
patient split boundaries. It requires the RTX host; the attempted SSH check at
`192.168.68.69:22` was refused, so no full-image run was launched.

This result does not show that a 75% development specificity is impossible. It
shows only that this particular feature-head family and its score fusion do
not advance it. All values in this report are retrospective development
metrics, not clinical performance claims or certification evidence.

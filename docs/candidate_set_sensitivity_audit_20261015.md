# Candidate-Panel by Patient-Split Sensitivity Audit

## Research question

Does the nested patient-grouped operating-point estimate remain stable when
the candidate score panel changes, and does that contrast persist across
different patient-group partitions?

## Data and procedure

The analysis uses the repeatedly inspected 981-eye, 928-patient pooled
development cohort (385 gradable nonreferable eyes). The same five-outer /
four-inner patient-grouped selector was run on two panels using fixed integer
RNG seeds 20261015--20261019. These date-shaped values are seed identifiers,
not calendar dates. The first is the original split; four additional seeds
were run to assess split sensitivity. Both panels include the fixed fused
recipe.

- **Oracle-contributor shortlist:** 11/1/7 total columns for DDR/DeepDRiD/DRTiD.
  Full-cohort outcome inspection selected channels that previously added at
  least one clearance. This is explicitly label-screened.
- **All complete aligned columns:** 44/40/119 total columns. This includes
  every complete, finite, nonconstant numeric score column aligned to each
  source in the saved CSV artifact library; ID, label, quality, and action
  fields are excluded; exact duplicate vectors are collapsed. Rank transforms
  are fit within training folds.

The second panel avoids label-based channel inclusion in these reruns, but the
artifact library and cohort were historically exposed. The five splits reuse
the same cohort and their training sets overlap. Across-seed SD/ranges are
descriptive split-sensitivity summaries, not confidence intervals or
independent replications.

## Seed-level results

| Seed | Oracle shortlist TN / 385 | Specificity | Misses | All aligned TN / 385 | Specificity | Misses | Difference (pp, all - oracle) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20261015 | 189 | 49.09% | 2 | 221 | 57.40% | 3 | +8.31 |
| 20261016 | 227 | 58.96% | 4 | 211 | 54.81% | 5 | -4.16 |
| 20261017 | 224 | 58.18% | 1 | 215 | 55.84% | 1 | -2.34 |
| 20261018 | 232 | 60.26% | 3 | 218 | 56.62% | 7 | -3.64 |
| 20261019 | 234 | 60.78% | 3 | 219 | 56.88% | 5 | -3.90 |

| Panel | Mean specificity | SD across seeds (pp) | Range | Mean TN | Mean misses | Zero-miss runs |
|---|---:|---:|---:|---:|---:|---:|
| Oracle-contributor shortlist | 57.45% | 4.79 | 49.09--60.78% | 221.2 | 2.6 | 0/5 |
| All complete aligned columns | 56.31% | 1.01 | 54.81--57.40% | 216.8 | 4.2 | 0/5 |

The mean paired contrast (all aligned minus oracle shortlist) is -1.14
percentage points, ranging from -4.16 to +8.31 points. All aligned has lower
specificity in four seeds and higher specificity in one. Neither panel reaches
zero misses in any of the five runs. The favorable initial seed reverses in
four alternate partitions; thus the single-seed 49.09% versus 57.40% contrast
is not a stable panel effect and does not improve the zero-miss point.

Under the descriptive constraint of fewer than five referable misses, the
highest specificity among the ten panel-by-seed outcomes is 60.78% (234/385
true negatives, three misses), from the oracle-contributor shortlist at seed
20261019. This maximum was selected after examining all ten outcomes and is
not a prespecified estimate or a new-patient performance claim. It must remain
distinct from the 70.13% full-cohort score-policy replay, which has zero
observed misses but uses cut points and rank references derived from that same
exposed cohort.

## Interpretation and novelty boundary

This is a project-specific empirical finding about the interaction between
candidate-panel construction and patient-fold realization in an exposed
multi-source DR score library. It is not a new learning algorithm, a
confidence interval for future performance, a causal estimate of candidate
count, or clinical validation. The nested-CV and model-selection-bias
principles are established. A broader novelty claim requires systematic
prior-art review and replication using a cohort and candidate library that
have not influenced this project.

The historical 70.13% score-policy replay (270/385, zero observed misses) is a
different same-cohort, full-cohort-selected evidence object. It is not the
output of either nested selector and must not be presented as their
generalization estimate. Neither nested panel produced a zero-miss run.

## Reproduction

Five run files per panel are summarized by:

```sh
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.summarize_candidate_panel_sweep
```

The machine-readable summary is
`results/candidate_panel_seed_sweep_20261015.json`. Each component run JSON
records channel names, fold selections, panel fingerprints, and source-file
SHA-256 values. Seed 20261015 component files retain the `candidate_set_contamination_*`
names; seeds 20261016--20261019 use `candidate_panel_<mode>_<seed>.json`.

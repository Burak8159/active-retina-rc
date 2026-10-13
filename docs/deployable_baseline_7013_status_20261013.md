# ActiveRetina 70.13 Baseline and Deployment Gate

## Frozen reference

The exposed pooled development recipe is retained unchanged as the reference:

- 270 true negatives out of 385 gradable nonreferable cases: **70.13% specificity**.
- 0 observed referable misses, 0 ungradable misses, and 100% binary action coverage on the same 981-case inspected pool.
- This is a retrospective, repeatedly inspected development result. Its feature blend, rank reference distributions, and thresholds were selected from this cohort; it is not an independent estimate of future performance.

The deterministic scorer in `src/active_retina/exposed_expert_policy.py` and `results/deployable_exposed_baseline_7013_20261013_with_components/policy.json` reproduce that recipe from its component score inputs. The replay matches the frozen counts exactly. This is a **fixed score-policy replay**, not a deployable raw-image model: it requires outputs from three component models per dataset, and several exact source checkpoints are not present in this checkout. The included artifact is explicitly marked `EXPOSED_REPLAY_ONLY_NOT_CLINICALLY_VALIDATED`.

## Patient-grouped development checks

Calibration folds use patient IDs from the deduplicated master manifest. Both are diagnostic because the model family and validation cohort have already been inspected.

| Procedure | Specificity | TN / negatives | Misses | Action coverage |
|---|---:|---:|---:|---:|
| Frozen same-cohort replay | 70.13% | 270 / 385 | 0 | 100% |
| Fixed recipe, 5-fold patient-disjoint calibration | 69.61% | 268 / 385 | 3 | 100% |
| Nested patient-fold blend-weight selection | 64.42% | 248 / 385 | 5 | 100% |
| Nested blend weights plus calibration-only safety-margin selection | 65.71% | 253 / 385 | 6 | 100% |
| Post-hoc source-specific margin frontier | **63.12%** | **243 / 385** | **0** | **100%** |

The nested procedures did not reach 75% and did not preserve zero misses on held-out folds. The post-hoc margin frontier reaches the relaxed 60% development number with zero observed misses, using extra rank-score margins of 0.005 for DDR, 0.05 for DeepDRiD, and 0.02 for DRTiD. **Those margins were selected using the held-out folds themselves**, so 63.12% is a diagnostic upper point on this inspected cohort, not an unbiased estimate, a deployable threshold, or a successful independent test. The reproducible diagnostic is explicitly labeled as post-hoc in `src/active_retina/posthoc_zero_miss_margin_sweep.py` and `results/posthoc_zero_miss_margin_sweep_20261013.json`.

## Deployment gate

Do not label the 70.13 scorer as a fully deployable model or a clinical/autonomous diagnostic system. To make the computation runnable end to end, the project still needs the exact component checkpoints, preprocessing contracts, and an image-to-score inference wrapper for each source, followed by a new patient-disjoint evaluation. The current nested result gives no support for a 75% specificity / zero-miss / 100%-coverage claim. No threshold or blend should be selected from the held-out fold results above.

Machine-readable outputs:

- `results/deployable_exposed_baseline_7013_20261013_with_components/summary.json`
- `results/crossfit_expert_calibration_20261013_corrected/summary.json`
- `results/nested_crossfit_expert_weights_20261013/summary.json`
- `results/nested_crossfit_expert_weights_margin_20261013/summary.json`

## Research profile registry

Both results are registered as distinct research profiles. List them with:

```bash
PYTHONPATH=src python -m active_retina.baseline_profiles --list
```

`70.13-replay` can replay an individual case only from precomputed component
scores and the explicit policy artifact. This is not image-to-diagnosis
deployment: exact component checkpoints and their inference/preprocessing
wrappers are not all present. Example:

```bash
PYTHONPATH=src python -m active_retina.baseline_profiles \
  --profile 70.13-replay --source DDR \
  --features-json case_component_scores.json \
  --policy-json results/deployable_exposed_baseline_7013_20261013_with_components/policy.json
```

`63.12-posthoc` is exposed in the same registry for benchmark comparison, but
is deliberately blocked for case inference because its margins were selected
using held-out labels. The clinical station does not load either profile.

For a local research-only HTTP service, run:

```bash
PYTHONPATH=src python -m active_retina.baseline_profiles --serve --host 127.0.0.1 --port 8766
```

- `GET /profiles` lists both entries and their evidence status.
- `POST /score` accepts `profile_id`, `source`, and precomputed `features`; only
  `70.13-replay` is eligible, and its response is explicitly research-only.
- `GET /diagnostics/63.12-posthoc` returns the aggregate inspected-cohort
  diagnostic. It is not a patient-level prediction endpoint.

The 70.13 policy JSON and the aggregate 63.12 report are bundled under
`configs/research_baselines/` for reproducible local operation. The service
does not accept image files and is not a clinical product or diagnosis API.

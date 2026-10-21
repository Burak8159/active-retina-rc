# ActiveRetina-RC replay reproduction package

## Scope

This package exposes two deliberately different audit levels. The clean-clone default checks the internal consistency of tracked aggregate summaries. The optional full replay executes the frozen policy against per-eye component-score rows, but those inputs are not distributed here. Neither mode reproduces image preprocessing, component-model inference, or model training, and neither is independent clinical validation. The reported 70.13% remains a same-cohort development replay.

## Run

Use Python 3.12 and install the pinned audit and test dependencies in a clean environment:

```sh
python3.12 -m venv .venv-paper-audit
. .venv-paper-audit/bin/activate
python -m pip install -r paper/reproducibility/requirements.txt
python paper/audit_submission_evidence.py --root . --mode summary --output paper/reproducibility/audit_report.json
python -m pytest -q tests
```

On Windows, activate the environment with `.venv-paper-audit\\Scripts\\Activate.ps1` instead. A successful clean-clone run emits `PASS_TRACKED_AGGREGATE_INTEGRITY_ONLY_NOT_REPLAY` and hashes the tracked summary files. In addition to the score-policy headline aggregates, it cross-checks the MMRDR validation trace against its tracked epoch history and run card, and verifies the official-test confusion counts, grade totals, sensitivity and specificity, their image-row Wilson intervals, and associated hashes. These checks establish internal consistency of the included aggregate records; they do not independently reconstruct outcomes from images or row-level observations. The focused tests include tamper-rejection cases, but do not reproduce model inference.

The row-level replay is an optional local operation:

```sh
python paper/audit_submission_evidence.py --root . --mode full-replay
```

It requires the two local evidence files below. If they are absent, the command fails closed. A successful run recomputes the actions and scores from supplied component-score rows and checks them against saved summaries. It is still not raw-image reproduction.

The local replay was rerun on 2026-10-19 and produced a byte-identical result to `local_full_replay_audit_20261017.json` (SHA-256 `40b916b12c598c02faf48d52e768c4e5ab923bf54a833b4fa9188624e7a74a4b`). The same aggregate-only output is retained as [`local_full_replay_audit_20261019.json`](local_full_replay_audit_20261019.json) with that SHA-256. Its per-source counts and headline outcomes reproduce 270/385 true negatives (70.1299%) and zero observed referable/ungradable clearances on the supplied exposed rows. This confirms deterministic replay of the same local inputs, not a new or independent validation result, and it is not a substitute for replaying the omitted row-level inputs from a clean clone.

Regression tests now reconcile every row in the manuscript's three-source replay table to this aggregate receipt and every row in the MMRDR post-hoc frontier tables to the hash-checked aggregate test-score diagnostic. The latter remains a searched, post-hoc description: its maxima are optimistically selected on the inspected test scores, no inferential interval is claimed for the searched envelope, and the export does not permit patient-clustered analysis. These source/table checks improve transcription auditability; they do not supply omitted raw predictions or reproduce model inference from images.

## Patient-clustered uncertainty for the exposed replay

To account descriptively for the 21 additional negative eyes contributed by bilateral DeepDRiD patients, run:

```sh
python paper/reproducibility/bootstrap_exposed_replay_specificity.py \
  --scores results/deployable_exposed_baseline_7013_20261013_with_components/validation_replay.csv \
  --manifest results/autonomy_route_three_dataset_manifest_20261011.csv \
  --output paper/reproducibility/exposed_replay_patient_cluster_bootstrap_20261019.json
```

The command resamples patient clusters with replacement within each source, keeps each selected patient's eligible negative-eye outcomes together, and reports a 50,000-replicate percentile interval. The checked aggregate receipt reports 364 negative-patient clusters across 385 negative eyes and a 95% interval of 65.63--74.55% for the 70.13% specificity. A regression test independently reconstructs the pooled draws from the toy cohort and checks the pooled interval and median separately from the pairwise source contrasts. For context only, treating eyes as independent gives a Wilson interval of 65.38--74.48%. The close intervals indicate little change from clustering in this particular cohort; neither interval adjusts for the policy having been selected on the exposed cohort, and neither supports a population or clinical-risk claim. The source score rows and manifest are local inputs and are not included in the public bundle; the script emits aggregate counts and SHA-256 hashes, never identifiers.

The same receipt reports source-specific patient-cluster intervals and exploratory pairwise source contrasts. DeepDRiD minus DDR specificity is 33.03 percentage points (95% percentile interval 25.24--40.29), while DeepDRiD minus DRTiD is 31.15 points (22.05--40.16); DDR minus DRTiD is -1.88 points (-12.31--8.54). Equal-source reweighting changes the same-cohort point estimate to 76.68%, with a source-stratified patient-cluster bootstrap percentile interval of 72.93--80.32%; the observed negative-eye mixture gives 70.13%. These conditional intervals do not adjust for cohort exposure or policy selection, and the contrasts do not identify why source rates differ. The 6.55-point difference is arithmetic reweighting, not model improvement or external validation. Equal-source weights are not asserted to represent a clinic. The receipt also reports the source-specific rates and their convex-mixture range; that range is not a confidence interval.

## Paired replay-versus-grouped action audit

Run the aggregate-only paired comparison with the local score rows and patient manifest:

```sh
PYTHONPATH=src python paper/reproducibility/paired_replay_crossfit_audit.py \
  --replay results/deployable_exposed_baseline_7013_20261013_with_components/validation_replay.csv \
  --manifest results/autonomy_route_three_dataset_manifest_20261011.csv \
  --output paper/reproducibility/paired_replay_crossfit_audit_20261020.json
```

The analysis requires pandas, NumPy, the local replay CSV, and the source-patient manifest. It reruns the five-fold grouped calibration, joins held-out actions to replay one-to-one, and fails closed on duplicate eyes, patient/source/label mismatch, or disagreement with the saved 270/385 replay and 268/385 grouped totals. The paired patient bootstrap separately resamples source--patient clusters within each endpoint stratum (gradable negatives; referable or ungradable eyes) and reports only aggregate transitions, intervals, and input hashes; it does not save case identifiers. Here, specificity differs by -0.52 percentage points (70.13% replay versus 69.61% grouped; conditional percentile 95% interval -1.31 to 0.00). On required cases, replay has 0/596 misses and grouped calibration has 3/596, a +0.50-point paired missed-case-rate difference (conditional percentile 95% interval 0.00 to 1.17; 574 positive-or-ungradable source--patient clusters). The latter interval includes zero. Both intervals condition on fixed predictions from an exposed cohort and exclude model/candidate/policy and calibration/threshold-selection uncertainty; neither is confirmatory or external validation. Row-level inputs and derived decisions remain local pending rights review.

## Required local evidence files

The clean Git checkout supports `--mode summary`. The optional `--mode full-replay` additionally requires:

- `results/deployable_exposed_baseline_7013_20261013_with_components/policy.json`
- `results/deployable_exposed_baseline_7013_20261013_with_components/validation_replay.csv`
- the five tracked `summary.json` files named in `paper/audit_submission_evidence.py`

These files are local, untracked, derived policy/score artifacts. They are excluded pending confirmation of redistribution rights for per-eye scores, labels, and reference vectors. Do not copy them into a public repository unless the applicable dataset terms and any access conditions permit it. The aggregate mode does not silently claim to replace this row-level replay.

## Reproduction boundaries

The replay CSV contains precomputed component scores and the policy contains empirical rank-reference values and thresholds. The script executes the policy against those rows; it cannot regenerate those component scores from images. The model checkpoints, complete preprocessing and inference commands, exact dataset release snapshots, and original cross-validation prediction files are not all included. Consequently, this is a deterministic replay package, not a full computational reproduction package.

The source datasets remain at their official distribution channels. Do not add images, restricted metadata, or derived per-eye outputs to the public repository until the release checklist's rights review is resolved. The repository's root `LICENSE` currently says **All rights reserved**; it does not grant an MIT license or other general reuse permission. That repository license also does not override third-party dataset terms.

## Controlled zero-miss selection simulation

Run the deterministic synthetic mechanism analysis with:

```sh
python paper/reproducibility/zero_miss_panel_simulation.py --seed 20261016 --replicates 2000
```

The script writes `paper/reproducibility/zero_miss_panel_simulation_results.json`. Every candidate channel has the same Gaussian class separation. A per-channel threshold is set to the minimum development-positive score (asserting zero misses by construction); the channel with greatest development specificity is selected and evaluated on a separate generated cohort. The saved output includes candidate-panel size, channel correlation, cohort sizes, 2,000-replicate summaries, simulation quantiles, and a Wilson interval for the Monte Carlo zero-test-miss frequency. These are properties of the specified simulation only, not clinical estimates, confidence intervals for patients, or reproduction of the retinal audit. The simulation was motivated after inspection of the project results and is exploratory rather than preregistered.

The JSON also includes an exact beta-function reference for independent, identically distributed channels when selection uses the candidate with maximum **true** specificity (equivalently, largest threshold under a shared strictly increasing negative-score CDF). It is a distinct idealized selector, not the finite-sample specificity selector simulated above. The tests verify both the single-channel limiting case and selected multi-channel values.

## Optional local MMRDR prediction-export integrity audit

If the private local prediction export is available, run:

```sh
python paper/reproducibility/audit_mmrdr_prediction_export.py \
  --prediction-export results/mmrdr_uwf_resnet50_official_test_20261015/test_predictions.csv \
  --output paper/reproducibility/local_mmrdr_prediction_export_audit_20261019.json
```

The saved 2026-10-19 receipt records the result of this local audit. The audit checks the export SHA-256 against the run card, row/label/action consistency, frozen threshold consistency, official row count, confusion counts, grade-stratified aggregates, and sensitivity/specificity Wilson intervals. It emits aggregate JSON only. The export itself is deliberately excluded from the review bundle; this verifies local prediction-export integrity and arithmetic, **not** image preprocessing, inference, checkpoint identity, training, or independent validation. Synthetic fail-closed cases are covered by `tests/test_mmrdr_prediction_export_audit.py`; `tests/test_mmrdr_official_test_receipt.py` ties the aggregate-only receipt to the run card and checks that it contains no image or patient identifiers.

## Post-hoc MMRDR score-ranking diagnostic

The local frozen prediction export can also be summarized with:

```sh
PYTHONPATH=src python -m active_retina.analyze_mmrdr_test_frontier \
  --prediction-export results/mmrdr_uwf_resnet50_official_test_20261015/test_predictions.csv \
  --output paper/reproducibility/mmrdr_test_score_diagnostic_20261019.json \
  --roc-display-csv paper/reproducibility/mmrdr_roc_display_aggregate_20261019.csv
```

This hash-checks the export and emits aggregate ROC-AUC, average precision, referral rates by grade at the locked threshold, a retrospective sensitivity-constrained specificity envelope, and a fixed 1\%-FPR sampled ROC display grid. The CSV contains only aggregate coordinates (plus the conventional origin), not image identifiers, scores, or thresholds. The official test labels had already been inspected when this diagnostic was run. Its alternative cutoffs and ROC curve are post-hoc descriptions only: they must not be used to select, tune, repair, or promote the model. Patient identifiers are absent, so no patient-clustered uncertainty is reported. Unit tests cover score ties, deterministic calculations, and hash-mismatch failure.

## Frozen MMRDR-UWF image-model test provenance

`mmrdr_uwf_official_test_run_card.json` records the historical ResNet-50 checkpoint digest, official test-manifest digest, evaluator-source digest, local prediction-export digest, threshold, and reported confusion counts. A provenance reconstruction also hashes the development manifest, cached feature file, feature extractor, and trainer, and records the trainer's checked-in defaults. The original launch command was not archived, so these defaults are not claimed to be the actual run arguments. The checkpoint is absent from the current checkout, so its historical digest cannot be reverified against the original bytes; the cache, official test manifest, images, and per-row predictions are also not in the public source archive. The run card is provenance metadata, not an executable reproduction bundle. The evaluator is `src/active_retina/evaluate_mmrdr_candidate.py`. The held-out result must not be used for threshold or model repair.

A development-only reconstruction check dated 2026-10-17 reran the checked-in defaults on the hash-verified development manifest and cached features. Historical versus reconstructed epoch-1 training loss was 2.7205 versus 10.5907; zero-miss validation specificity was 4.67% versus 0.50%, at thresholds -1.3501 versus 3.8018. The run completed 11 validation epochs and stopped before saving because pretrained backbone weights were unavailable locally. Official-test inputs and outputs were not read. This failed reproduction does not identify which missing run-specific arguments, runtime differences, or other provenance caused the mismatch; it is not a new model result and does not change the official-test evaluation.

The local manifest and prediction export can be joined without publishing row-level records:

```sh
python paper/reproducibility/audit_mmrdr_manifest_join.py \
  --test-manifest results/mmrdr_uwf_official_test_20261013.csv \
  --prediction-export results/mmrdr_uwf_resnet50_official_test_20261015/test_predictions.csv \
  --source-annotations external/MMRDR/MMRDR-UWF/UWF.csv \
  --output paper/reproducibility/local_mmrdr_manifest_prediction_join_audit_20261019.json
```

The aggregate-only receipt verifies a one-to-one image-ID set match and grade/label agreement for all 2,597 prediction/manifest rows, then joins those test images to the 10,404-row source annotation CSV and verifies grade agreement. The source CSV has columns `type`, `image`, `grade`, `lesion`, and `lr` (laterality), but no patient-link key. The derived manifest contains `patient_id` and `eye_id` columns, but both are empty for every row. Therefore, the joins verify row identity and labels, not patient grouping, and cannot support patient-clustered confidence intervals. The receipt omits identifiers and does not rerun inference. The source CSV, derived manifest, and prediction export are not included in the public review bundle; the receipt is aggregate-only.

## Post-hoc exact-byte train/test overlap audit

With the original local UWF images available, the exact-image audit can be rerun as follows:

```sh
python paper/reproducibility/audit_mmrdr_exact_duplicate_impact.py \
  --manifest results/mmrdr_uwf_manifest_20261013.csv \
  --prediction-export results/mmrdr_uwf_resnet50_official_test_20261015/test_predictions.csv \
  --feature-cache results/mmrdr_uwf_features_20261013.npz \
  --root . \
  --output paper/reproducibility/mmrdr_exact_duplicate_impact_20261019.json
```

The 2026-10-19 aggregate receipt reports eight exact-byte duplicate groups among 10,404 local manifest rows. Two groups cross the dataset's official train/test partitions; both test images are positive, their training copies are in the internal model-fit split, and both were referred by the frozen classifier. Removing those two test rows post hoc leaves five false negatives among 1,286 positives and unchanged specificity (262/1,309; 20.02%). The primary manuscript result retains the complete official test rows. This audit changes the split-independence interpretation, not the reported model or threshold. The test results were already inspected, so the filtered counts are descriptive, not a new confirmatory estimate. Matching is on original bytes only; it cannot find resized/recompressed near-duplicates or determine participant identity. The script writes aggregate counts and input hashes only; row identifiers and per-image actions remain local. Since the underlying data are not part of this package, a clean clone cannot rerun this image-level audit.

## MMRDR validation epoch-selection trace

`../submission/mmrdr_validation_epoch_trace_2026-10-16.csv` is generated from the tracked aggregate history with:

```sh
python paper/reproducibility/export_mmrdr_validation_trace.py \
  results/mmrdr_uwf_cached_head_20261013/summary.json \
  paper/submission/mmrdr_validation_epoch_trace_2026-10-16.csv
python -m unittest tests.test_mmrdr_validation_trace -v
```

The exporter checks count arithmetic and verifies that the recorded selected epoch is a zero-miss validation epoch with maximum specificity among the recorded zero-miss epochs. The 26-row trace shows epoch 20 selected at 123/600 true negatives (20.50% specificity), with zero validation misses. This is an aggregate validation-selection trace only: it does not establish that selection was prespecified, reproduce image inference, recover the original launch arguments or checkpoint, or provide independent/external validation. The held-out official-test result remains five false-negative image rows and 20.02% specificity.

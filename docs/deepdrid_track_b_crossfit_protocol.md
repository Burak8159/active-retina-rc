# DeepDRiD ActiveRetina-RC Cross-Fitted Protocol

Date: 2026-10-03

Status: `DATA_READY_METRICS_BLOCKED_ON_CROSSFIT_PREDICTIONS`

DeepDRiD is now the primary available dataset for the upgraded ActiveRetina-RC flagship. It supports paired same-eye two-view adaptive acquisition, calibrated stopping, and defer behavior. It does not support four-field field-selection or non-myopic multi-step planning claims.

## Mapping Policy

Run both mapping assumptions end to end:

- Visual-audit mapping: `configs/deepdrid.yaml`
  - `suffix_1 = optic_disc`
  - `suffix_2 = macula`
  - status: `INFERRED_NOT_OFFICIAL_VISUAL_AUDIT`
- Swapped sensitivity mapping: `configs/deepdrid_swapped_mapping_sensitivity.yaml`
  - `suffix_1 = macula`
  - `suffix_2 = optic_disc`
  - status: `SENSITIVITY_ANALYSIS_SWAPPED_NOT_OFFICIAL`

If conclusions change materially between these two branches, field mapping remains a blocker pending official author confirmation. Do not average the two branches.

## Locked Split

Both mapping branches use the same immutable patient split seed: `20261003`.

Current split counts:

- train: 279 patients
- calibration: 40 patients
- validation: 40 patients
- sealed test: 40 patients
- total: 399 patients

The sealed test split must not be used for model selection, threshold choice, mapping choice, or architecture changes.

## Required Prediction File

The evaluator requires out-of-fold/cross-fitted predictions. It deliberately refuses ordinary in-sample predictions.

Required files:

- visual-audit branch: `results/deepdrid_crossfit_predictions.csv`
- swapped branch: `results/deepdrid_swapped_mapping_crossfit_predictions.csv`

Required columns:

- `patient_id`
- `eye_id`
- `view_set`
- `grade`
- `referable_dr`
- `referable_score`
- `fold_id`
- `prediction_scope`

Allowed `prediction_scope` values:

- `oof`
- `out_of_fold`
- `crossfit`
- `cross_fitted`

Required `view_set` rows per eye:

- `macula`
- `macula+optic_disc`

## Evaluated Quantities

The evaluator computes:

- macula-only referable DR AUROC/AUPRC/Brier
- macula+optic-disc AUROC/AUPRC/Brier
- cross-fitted hindsight STOP oracle upper bound
- mean Brier loss reduction from the second view
- fraction of eyes where the second view improved hindsight Brier loss

The hindsight STOP oracle may use true labels only for retrospective analysis of cross-fitted predictions. It is not a deployable policy and must not be described as learned stopping.

The flagship policy layer to add next computes:

- calibrated `STOP` thresholds,
- calibrated `ACQUIRE_SECOND_VIEW` thresholds,
- calibrated `DEFER` thresholds,
- image savings at target missed-referable-DR risk levels,
- sealed-test cost-risk curves.

## Data Acquisition

The public DeepDRiD v1.1 archive was downloaded from Zenodo record `8248825`.

Verified archive:

- path: `external/DeepDRiD-v1.1.zip`
- expected MD5: `3379e2fd7a2dd398545a67148420a5d3`
- observed MD5: `3379e2fd7a2dd398545a67148420a5d3`
- extracted root: `external/DeepDRiD_v1_1/deepdrdoc-DeepDRiD-56d8af7`

Regular fundus local inventory:

- total regular fundus images in archive: 2000
- labeled train/validation manifest rows: 1589
- manifest rows with existing local image files: 1589
- missing labeled train/validation image files: 0

## Current Blocker

Full DeepDRiD regular fundus images are locally available and schema/image integrity has been verified for labeled train/validation rows. The current metric outputs are expected to be:

- `BLOCKED_CROSSFIT_PREDICTIONS_MISSING`

This is a data-ready state, not empirical evidence. The next step is generating out-of-fold predictions for `macula`, `optic_disc`, and `macula+optic_disc` under both mapping assumptions.

## Commands

```bash
PYTHONPATH=src python3 -m active_retina.prepare_deepdrid --config configs/deepdrid.yaml
PYTHONPATH=src python3 -m active_retina.prepare_deepdrid --config configs/deepdrid_swapped_mapping_sensitivity.yaml
PYTHONPATH=src python3 -m active_retina.make_patient_splits --config configs/deepdrid.yaml
PYTHONPATH=src python3 -m active_retina.make_patient_splits --config configs/deepdrid_swapped_mapping_sensitivity.yaml
PYTHONPATH=src python3 -m active_retina.make_multiview_manifest --manifest results/deepdrid_manifest.csv --output results/deepdrid_multiview_manifest.jsonl
PYTHONPATH=src python3 -m active_retina.make_multiview_manifest --manifest results/deepdrid_swapped_mapping_manifest.csv --output results/deepdrid_swapped_mapping_multiview_manifest.jsonl
PYTHONPATH=src python3 -m active_retina.evaluate_deepdrid_track_b --config configs/deepdrid.yaml
PYTHONPATH=src python3 -m active_retina.evaluate_deepdrid_track_b --config configs/deepdrid_swapped_mapping_sensitivity.yaml
```

## Flagship Boundary

DeepDRiD can support the ActiveRetina-RC flagship claim if risk-controlled adaptive stopping saves images without violating the target missed-referable-DR risk on sealed patients.

Only MFIDDR or another true multi-step same-eye dataset can decide:

- different eyes require different next retinal fields
- 1/2/3/4-field subset value
- greedy versus non-myopic multi-step planning value
- final `GO-CCG`, `GO-MYOPIC`, or empirical `NO-GO` for four-field CCG

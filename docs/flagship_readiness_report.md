# ActiveRetina-RC Flagship Readiness Report

Date: 2026-10-03

Status: `PIPELINE_READY_BASELINE_EVIDENCE_AVAILABLE`

This report summarizes the executable baseline produced after the flagship upgrade. The current model family is a dependency-light image-feature logistic baseline, used to verify the end-to-end pipeline and provide preliminary evidence. It is not the final pretrained encoder needed for paper-level performance claims.

## Current Decision

`PRELIMINARY_PIPELINE_GO`

The pipeline is ready for the stronger pretrained-encoder run. The preliminary baseline shows paired-view value and viable calibrated policy behavior under both DeepDRiD mapping assumptions, but final flagship claims remain pending a stronger encoder and frozen final reporting.

## Visual-Audit Mapping Paired-View Value

- Status: `ACTIVE_RETINA_RC_PAIRED_VIEW_VALUE_EVALUATED`
- Mapping status: `INFERRED_NOT_OFFICIAL_VISUAL_AUDIT`
- Paired eyes: 792
- First-view AUROC: 0.8023; Brier: 0.1813
- Two-view AUROC: 0.8095; Brier: 0.1778
- AUROC gain from second view: 0.0072
- Cross-fitted hindsight AUROC upper bound: 0.8638
- Fraction of eyes where second view reduced Brier loss: 0.5253

## Swapped Mapping Sensitivity Paired-View Value

- Status: `ACTIVE_RETINA_RC_PAIRED_VIEW_VALUE_EVALUATED`
- Mapping status: `SENSITIVITY_ANALYSIS_SWAPPED_NOT_OFFICIAL`
- Paired eyes: 792
- First-view AUROC: 0.7720; Brier: 0.1954
- Two-view AUROC: 0.8095; Brier: 0.1778
- AUROC gain from second view: 0.0375
- Cross-fitted hindsight AUROC upper bound: 0.8692
- Fraction of eyes where second view reduced Brier loss: 0.5606

## Visual-Audit Mapping Calibrated Policy

- Status: `ACTIVE_RETINA_RC_POLICY_EVALUATED`
- Decision: `PRELIMINARY_PIPELINE_GO`
- Calibration eyes: 80
- Validation eyes: 78
- Sealed-test eyes: 80

| Target risk | Calibration status | Sealed-test missed-risk | Mean images | Savings vs two-view | Defer rate |
|---:|---|---:|---:|---:|---:|
| 0.01 | `CERTIFIED_ON_CALIBRATION` | 0.0312 | 1.0625 | 0.4688 | 0.0000 |
| 0.02 | `CERTIFIED_ON_CALIBRATION` | 0.0312 | 1.0625 | 0.4688 | 0.0000 |
| 0.05 | `CERTIFIED_ON_CALIBRATION` | 0.0312 | 1.0625 | 0.4688 | 0.0000 |
| 0.10 | `CERTIFIED_ON_CALIBRATION` | 0.0938 | 1.0000 | 0.5000 | 0.0000 |

## Swapped Mapping Sensitivity Calibrated Policy

- Status: `ACTIVE_RETINA_RC_POLICY_EVALUATED`
- Decision: `PRELIMINARY_PIPELINE_GO`
- Calibration eyes: 80
- Validation eyes: 78
- Sealed-test eyes: 80

| Target risk | Calibration status | Sealed-test missed-risk | Mean images | Savings vs two-view | Defer rate |
|---:|---|---:|---:|---:|---:|
| 0.01 | `CERTIFIED_ON_CALIBRATION` | 0.0000 | 1.0125 | 0.4938 | 0.0000 |
| 0.02 | `CERTIFIED_ON_CALIBRATION` | 0.0000 | 1.0125 | 0.4938 | 0.0000 |
| 0.05 | `CERTIFIED_ON_CALIBRATION` | 0.0625 | 1.0250 | 0.4875 | 0.0125 |
| 0.10 | `CERTIFIED_ON_CALIBRATION` | 0.0625 | 1.0250 | 0.4875 | 0.0125 |

## What Is Ready

- DeepDRiD local data audit and immutable patient splits.
- Mapping sensitivity branch.
- Cross-fitted prediction CSV generation.
- Paired-view value evaluator.
- Calibrated `STOP / ACQUIRE_SECOND_VIEW / DEFER` policy evaluator.
- Bootstrap CIs in metrics JSON outputs.
- Explicit evidence boundaries separating executable baseline evidence from final pretrained-encoder claims.

## Still Required For The Actual Paper

- Replace the dependency-light baseline with a strong pretrained retinal/image encoder.
- Re-run both mapping branches with the stronger encoder.
- Add external public DR robustness datasets where legally accessible.
- Freeze policy families before final sealed-test reporting.
- Generate publication plots and IEEEtran manuscript tables from frozen outputs.

## Evidence Boundary

No MFIDDR, four-field CCG, non-myopic planning, clinical deployment, or official DeepDRiD suffix-mapping claim is made here.

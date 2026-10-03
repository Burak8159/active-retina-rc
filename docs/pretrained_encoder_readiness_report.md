# ActiveRetina-RC Pretrained Encoder Readiness Report

Date: 2026-10-03

Status: `METHOD_ONLY_WEAK_EVIDENCE_MAPPING_SENSITIVITY_BLOCKER`

Model: frozen ImageNet-pretrained `resnet50.a1_in1k` features with regularized logistic heads.

This is stronger than the dependency-light baseline and is now the main DeepDRiD evidence artifact. It is still not a completed paper package because external robustness datasets and final manuscript figures remain pending.

## Paired-View Value

| Mapping branch | First-view AUROC | Two-view AUROC | AUROC gain | First Brier | Two-view Brier | Hindsight AUROC | Second view helped |
|---|---:|---:|---:|---:|---:|---:|---:|
| visual_audit | 0.8868 | 0.9042 | 0.0174 | 0.1616 | 0.1422 | 0.9503 | 0.5467 |
| swapped_sensitivity | 0.8649 | 0.9042 | 0.0392 | 0.1670 | 0.1422 | 0.9457 | 0.5707 |

## Calibrated Policy With Specificity Floor

| Mapping branch | Target risk | Status | Test missed-risk | Mean images | Savings | Defer rate | Sensitivity | Specificity |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| visual_audit | 0.01 | `CERTIFIED_ON_CALIBRATION` | 0.0312 | 1.0250 | 0.4875 | 0.0000 | 0.9688 | 0.5833 |
| visual_audit | 0.02 | `CERTIFIED_ON_CALIBRATION` | 0.0312 | 1.0250 | 0.4875 | 0.0000 | 0.9688 | 0.5833 |
| visual_audit | 0.05 | `CERTIFIED_ON_CALIBRATION` | 0.0625 | 1.0250 | 0.4875 | 0.0000 | 0.9375 | 0.6042 |
| visual_audit | 0.10 | `CERTIFIED_ON_CALIBRATION` | 0.0625 | 1.0250 | 0.4875 | 0.0000 | 0.9375 | 0.6042 |
| swapped_sensitivity | 0.01 | `CERTIFIED_ON_CALIBRATION` | 0.1562 | 1.0250 | 0.4875 | 0.0000 | 0.8438 | 0.6250 |
| swapped_sensitivity | 0.02 | `CERTIFIED_ON_CALIBRATION` | 0.1562 | 1.0250 | 0.4875 | 0.0000 | 0.8438 | 0.6250 |
| swapped_sensitivity | 0.05 | `CERTIFIED_ON_CALIBRATION` | 0.1562 | 1.0125 | 0.4938 | 0.0000 | 0.8438 | 0.6667 |
| swapped_sensitivity | 0.10 | `CERTIFIED_ON_CALIBRATION` | 0.1562 | 1.0125 | 0.4938 | 0.0000 | 0.8438 | 0.6667 |

## Interpretation

- Pretrained features produce strong paired-view discrimination: two-view AUROC is about 0.904 in both mapping branches.
- The visual-audit branch supports a preliminary image-saving policy under the 0.10 target risk with a 0.50 specificity floor.
- The swapped branch fails the same sealed-test risk check under the specificity-constrained policy, so suffix mapping remains a material blocker for a field-named paper claim.
- A suffix-agnostic paired-view claim is possible, but the stronger IEEE/arXiv paper still needs mapping resolution or a mapping-robust policy family.

## Next Required Work

- Resolve DeepDRiD suffix mapping by official source or make the paper explicitly suffix-agnostic.
- Add at least one external public DR robustness dataset.
- Generate final plots and IEEEtran manuscript tables after the policy family is frozen.

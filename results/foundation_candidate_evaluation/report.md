# Foundation Candidate Smoke Evaluation

Correction (2026-10-11): historical `GO_EXTRA_SIGNAL_IMPROVES_AUTONOMY` rows below are superseded. The former max-coverage mixture was selected by test performance among 5,000 candidates. A validation-only rerun is recorded in `results/autonomy_extra_signal_validation_audit_20261011_clean/summary.json`; its exposed-test result is 83.0% specificity among 72.6% diagnostic decisions, zero observed misses, and 27.4% insufficient-evidence cases. Forcing those cases to `REFER` yields 33.9% binary specificity. No result here establishes full-binary autonomy or certification.

This report evaluates candidate representation-changing retinal/fundus models against the current three-output autonomy baseline. These are smoke studies, not final certification runs.

Current full-data baseline: zero false-negative clearances, 94.7% specificity among diagnostic decisions, 43.8% diagnostic decision rate.

| Run | Status | Positive/Test N | False Clearances | Specificity | Diagnostic Rate | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| RETFound_DR_IDRID_unbalanced_smoke | `GO_IMPROVED_THREE_OUTPUT_AUTONOMY` | 0/32 | 0 | 100.0% | 100.0% | invalid smoke: no positives |
| RETFound_DR_IDRID_balanced_smoke | `NO_GO_IMPROVED_THREE_OUTPUT_AUTONOMY` | 48/72 | 0 | 100.0% | 6.9% | safe smoke, compare coverage |
| FundusDRGrading_DINOv2_balanced_smoke | `NO_GO_IMPROVED_THREE_OUTPUT_AUTONOMY` | 48/72 | 0 | 20.0% | 58.3% | zero misses but poor specificity |
| FundusDRGrading_DINOv2_inverted_balanced_smoke | `NO_GO_IMPROVED_THREE_OUTPUT_AUTONOMY` | 48/72 | 0 | 0.0% | 0.0% | zero misses but poor specificity |
| ClementP/FundusDRGrading-resnet50_balanced_smoke | `NO_GO_IMPROVED_THREE_OUTPUT_AUTONOMY` | 48/72 | 0 | 0.0% | 0.0% | standard Transformers head exposes one scalar label; no valid policy |
| ClementP/FundusDRGrading-tf_efficientnet_b5_balanced_smoke | `NO_GO_IMPROVED_THREE_OUTPUT_AUTONOMY` | 48/72 | 0 | 0.0% | 44.4% | safe clearance fails on test negatives |
| ClementP/FundusDRGrading-tf_efficientnet_b5_full | `NO_GO_IMPROVED_THREE_OUTPUT_AUTONOMY` | 946/1435 | 0 | 0.0% | 0.0% | standalone policy cannot select a valid full validation rule |
| Augmented ensemble with ClementP EfficientNet B5 | `SUPERSEDED_TEST_SELECTED` | 946/1435 | 0 | 92.3% | 52.5% | exploratory only; former test-selected comparison |
| Augmented random-mixture max-coverage mode | `SUPERSEDED_TEST_SELECTED` | 946/1435 | 0 | 90.8% | 66.8% | invalid as an independent test claim; candidate was chosen using test metrics |
| Augmented random-mixture conservative mode | `SUPERSEDED_TEST_SELECTED` | 946/1435 | 0 | 93.1% | 54.4% | invalid as an independent test claim; candidate was chosen using test metrics |

## Conclusion

RETFound, DINOv2, and ClementP fundus classifiers remain evaluated as development signals. Direct replacement models did not improve the binary target. The corrected mixture audit and its limitations are summarized above; it does not support a full-binary GO.

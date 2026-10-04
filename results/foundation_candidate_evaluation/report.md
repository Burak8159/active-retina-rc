# Foundation Candidate Smoke Evaluation

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
| Augmented ensemble with ClementP EfficientNet B5 | `GO_EXTRA_SIGNAL_IMPROVES_AUTONOMY` | 946/1435 | 0 | 92.3% | 52.5% | useful as an added signal, not as a standalone policy |
| Augmented random-mixture max-coverage mode | `GO_EXTRA_SIGNAL_IMPROVES_AUTONOMY` | 946/1435 | 0 | 90.3% | 66.7% | current highest safe diagnostic coverage |
| Augmented random-mixture conservative mode | `GO_EXTRA_SIGNAL_IMPROVES_AUTONOMY` | 946/1435 | 0 | 93.1% | 54.4% | lower coverage but cleaner specificity margin |

## Conclusion

RETFound, DINOv2, and ClementP fundus classifiers are now connected to the autonomy evaluator. Direct replacement models did not improve the current baseline. The useful result is signal-level: adding the ClementP EfficientNet B5 score to the existing ensemble improved held-out diagnostic decision rate from 43.8% to 66.7% while preserving zero false-negative clearances and meeting the 90% specificity floor. A conservative augmented mode reaches 54.4% diagnostic decisions at 93.1% specificity. The learned fusion head did not pass because it introduced false-negative clearances, so the current product frontier should use the validation-selected augmented score-mixture policies rather than the learned fusion head.

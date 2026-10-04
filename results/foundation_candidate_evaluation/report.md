# Foundation Candidate Smoke Evaluation

This report evaluates candidate representation-changing retinal/fundus models against the current three-output autonomy baseline. These are smoke studies, not final certification runs.

Current full-data baseline: zero false-negative clearances, 94.7% specificity among diagnostic decisions, 43.8% diagnostic decision rate.

| Run | Status | Positive/Test N | False Clearances | Specificity | Diagnostic Rate | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| RETFound_DR_IDRID_unbalanced_smoke | `GO_IMPROVED_THREE_OUTPUT_AUTONOMY` | 0/32 | 0 | 100.0% | 100.0% | invalid smoke: no positives |
| RETFound_DR_IDRID_balanced_smoke | `NO_GO_IMPROVED_THREE_OUTPUT_AUTONOMY` | 48/72 | 0 | 100.0% | 6.9% | safe smoke, compare coverage |
| FundusDRGrading_DINOv2_balanced_smoke | `NO_GO_IMPROVED_THREE_OUTPUT_AUTONOMY` | 48/72 | 0 | 20.0% | 58.3% | zero misses but poor specificity |
| FundusDRGrading_DINOv2_inverted_balanced_smoke | `NO_GO_IMPROVED_THREE_OUTPUT_AUTONOMY` | 48/72 | 0 | 0.0% | 0.0% | zero misses but poor specificity |

## Conclusion

RETFound and DINOv2 candidates are now connected to the autonomy evaluator. On balanced smoke subsets, neither improves the current baseline. RETFound is very conservative; DINOv2 increases decisions but loses specificity. The next credible step is not declaring failure of foundation models, but running additional compatible checkpoints and then full validation/test only for candidates that pass a balanced smoke with positives.

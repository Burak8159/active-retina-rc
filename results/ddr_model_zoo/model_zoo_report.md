# DDR Model-Zoo Evaluation

This report converts the current DDR experiments into a paper-ready model-comparison block. It follows the structure of mature biomedical ML figures: multiple model families, balanced metrics, and score-distribution diagnostics rather than a single preferred run.

## Main Result

- Best overall discrimination model by MCC: `feature_mlp` with MCC `0.345`, AUROC `0.940`, AUPRC `0.970`, and specificity `17.8%` at its logged operating point.
- Best zero-miss model at the logged operating point: `feature_ensemble` with specificity `8.4%` and false negatives `0`.
- No evaluated every-image model currently satisfies `false_negative = 0` and `specificity >= 90%`.

## Paper Figure Package

- `results/ddr_model_zoo/model_zoo_metrics.csv`
- `results/ddr_model_zoo/model_zoo_metric_heatmap.svg`
- `results/ddr_model_zoo/feature_ensemble_score_distribution.svg`
- `results/ddr_model_zoo/feature_mlp_score_distribution.svg`

## Interpretation

The benchmark shows that high AUROC/AUPRC does not automatically translate into every-image autonomous diagnosis under a zero-miss clearance constraint. This is the central technical gap: the current models rank referable disease well, but their negative-class tail is not clean enough to clear every image autonomously at high specificity.

The practical paper claim should therefore emphasize risk-controlled selective autonomy now, and use the failed every-image experiments as a falsification analysis. The next model-development step should be an explicitly clearance-trained architecture with hard-positive mining, image-quality gating, and a validation objective that directly maximizes nonreferable clearance subject to zero positive clearances.

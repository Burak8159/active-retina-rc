# ActiveRetina-RC Flagship Evidence Layer

Status: `GO_DEEPDRID_ARXIV_BENCHMARK_CONDITIONAL_IEEE`

This pass upgrades the project from a short policy draft to a falsifiable benchmark package with a positive publication decision for a transparent arXiv methods/benchmark preprint. It does not claim clinical deployment readiness. IEEE journal submission remains conditional on a stronger DeepDRiD-only finite-sample risk-control and calibration path.

## Reproducible command

```bash
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.flagship_evidence_layer
```

## New artifacts

- `results/flagship_frontier.csv`
- `results/flagship_cost_frontier.csv`
- `results/flagship_view_value.csv`
- `results/flagship_mechanism.csv`
- `results/flagship_oracle_bounds.csv`
- `results/flagship_decision.json`
- `plots/flagship_risk_image_frontier.svg`
- `plots/flagship_risk_image_frontier.png`
- `plots/flagship_paired_view_value.png`
- `plots/flagship_mechanism_brier.png`
- `plots/flagship_cost_sensitivity.png`
- `results/formal_risk_control_summary.json`
- `results/formal_risk_control_selected.csv`
- `results/formal_risk_control_candidates.csv`
- `results/deepdrid_full_cv_predictions.csv`
- `results/deepdrid_full_cv_metrics.csv`
- `results/deepdrid_full_cv_policy.csv`
- `results/deepdrid_full_cv_folds.csv`

## Core scientific argument

The paper-level hypothesis is no longer "adaptive imaging saves images." That is too weak and is confounded by deferral.

The falsifiable hypothesis is:

> Selective paired-view acquisition is useful only if it improves the risk-coverage-cost frontier over first-view confidence triage under patient-disjoint validation.

## What the current evidence supports

- Paired views improve validation AUROC by 0.020 to 0.041 under the two suffix-mapping branches.
- Paired views improve validation Brier score by 0.009 to 0.036.
- At the strict 1% calibration risk / 0.50 specificity setting, acquisition lowers observed conditional missed-risk point estimates relative to confidence triage.
- The strongest mechanism is not uniform. The second view is most useful for moderate DR and high first-vs-two-view score-discordance groups.

## What the current evidence does not support

- It does not prove IEEE-level clinical safety.
- It does not prove acquisition dominance over confidence triage.
- It does not resolve the public DeepDRiD suffix-to-anatomy mapping.
- It does not repair the exposed historical test partition.
- It does not complete secondary five-class grading.

## Decision

`GO_DEEPDRID_ARXIV_BENCHMARK_CONDITIONAL_IEEE`

The work is now a GO for a reproducible DeepDRiD-only benchmark-style arXiv paper: it contains real paired-view predictions, five-fold patient-level cross-validation, binary referable-DR and five-class grading endpoints, inner-calibrated policy evaluation, risk-coverage-cost baselines, bootstrap uncertainty, mapping sensitivity, mechanism diagnostics, generated figures, provenance hashes, and exact calibration-set risk upper bounds.

The next legitimate upgrade toward an unqualified IEEE journal claim is not waiting for unavailable data; it is improving the DeepDRiD-only ML stack with prespecified risk control, stronger calibration, and complete secondary analysis.

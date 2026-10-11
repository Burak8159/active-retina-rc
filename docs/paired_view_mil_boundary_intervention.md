# Paired-View MIL Boundary Intervention

Date: 2026-10-11
Status: `RTX_DEVELOPMENT_RUN_COMPLETE; TARGET_NOT_MET`

## Diagnosis

The current three-source eye-level model combines image embeddings before its disease-risk head. The eye-level label is correct, but this early fusion can dilute a lesion signal that is visible mainly in one field. The prior attempts changed resolution, source balance, or grade-boundary sampling and did not improve zero-miss validation specificity. The next candidate therefore changes the label-to-view learning path rather than repeating those interventions.

## Implemented candidate

`train_mixed_view_bag_autonomy.py` now supports `--risk-pooling logmeanexp`:

- A shared disease head scores each view embedding.
- A masked, temperature-controlled log-mean-exp pools view logits into one eye-level risk score.
- The eye label supervises the pooled score only; no per-view labels are fabricated.
- The grade and image-quality heads retain the existing eye-level fused representation.
- Single-view eyes use the one valid score exactly; padded views receive no gradient.
- Existing fused-risk behavior remains the default for reproducibility.
- `--skip-test-evaluation` avoids loading/scoring the already-exposed development test images during candidate development.
- Validation diagnostics report the unchanged selected policy by dataset source and reference grade, making residual domain/boundary failures visible.

## Local verification

- Python compilation passed for the trainer and updated test module.
- Direct PyTorch checks passed: pooled values are finite, both valid views receive gradient, padded views receive zero gradient, and a single-view bag returns its valid logit exactly.
- Manifest view-count checks passed without opening image files.
- The repository environment does not have `pytest` installed, so the pytest module was not run here.

## RTX result

The one-epoch RTX gate run completed on the three-source patient-disjoint development manifest. At the validation-selected zero-miss threshold, specificity was 37.31% (144/386 gradable nonreferable eyes), with 0/463 referable eyes missed and 100% case-action coverage. The matched source-balanced baseline was 24.61% (95/386), a +12.70 percentage-point development gain. Test evaluation was skipped.

A warm-start continuation was stopped after its first epoch reduced zero-miss specificity to 28.24%. The saved gate-run checkpoint is retained on the RTX; do not continue this fine-tuning branch. The result remains far below the 80% specificity target, with especially weak grade-1 and cross-source performance. Full metrics and per-source diagnostics are recorded in `docs/rtx_logmeanexp_mil_result_20261011.md` and the local run artifacts.

This is development evidence only, not an untouched certificate, clinical validation, or achievement of `80/0/100`.

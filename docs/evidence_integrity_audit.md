# Evidence Integrity Audit

The previous results remain exploratory. They cannot support a certified
risk-control claim or an untouched final test claim.

## Findings and Corrections

- Previous cross-fitting pooled train, calibration, and validation patients.
  Each prediction excluded its own patient, but calibration labels influenced
  other fitted models. Test scores came from a differently trained model.
  The corrected trainer cross-fits only training patients and fits a single
  train-only model for calibration and validation predictions.
- Searching thresholds for empirical calibration error does not certify risk.
  The evaluator now labels these policies empirically selected and uncertified.
  Independent patient-level certification and selection correction remain open.
- The prior bootstrap sampled eyes independently. Both eyes from a patient
  must travel together; policy confidence intervals now resample patients.
- Undefined risk or specificity must not make a policy feasible. No decided
  positive patients now yields undefined risk, and undefined specificity fails.
- Previous test results were inspected while the policy family was changing.
  The original test manifest is preserved, but it is now an exposed exploratory
  holdout. Relabeling or shuffling the same patients cannot undo this exposure.
  A new independent confirmatory cohort is required for final claims.

## Current Decision

METHOD_ONLY_WEAK_EVIDENCE. Continue development. Neither FLAGSHIP_GO nor
risk certification is supported yet. DeepDRiD can test paired acquisition;
it cannot establish four-field non-myopic planning. View suffixes must remain
explicit sensitivity branches until their anatomical mapping is confirmed.

Existing encoder features can be reused. Corrected prediction files use new
filenames so original artifacts remain traceable. Test processing and evaluation
require an explicit flag and are disabled for development runs.

## Corrected Development Run

Both suffix-mapping branches were rerun using cached pretrained ResNet-50
features, five training-only patient folds, and a final train-only classifier.
Calibration and validation predictions come from that same fitted classifier.
There are 2,141 prediction rows in the primary branch and zero test rows.
The exploratory threshold grid uses 11 points per axis; the grid is not a
frozen final protocol. Patient-cluster bootstrap uses 2,000 replicates.

For the visually inferred branch, the policy selected at the 5% empirical
target has validation image savings of 39.74%, deferral of 20.51%, and
missed referable DR among decided referable eyes of 3.57%. The patient-bootstrap
95% interval for that risk is 0%-11.54%, too wide to substantiate a 5% claim.

The swapped branch has an exploratory zero-event policy at the 1%/2% targets
with 29.49% image savings. Its zero-width percentile-bootstrap risk interval
is a zero-event artifact, not proof of zero population risk. A finite-sample
upper risk bound with independent patient-level certification is required.

Artifacts: `results/deepdrid_train_only_development_predictions.csv`,
`results/deepdrid_swapped_train_only_development_predictions.csv`, and the
corresponding `*_development_policy.json` files. Both branches yield
`DEVELOPMENT_SIGNAL_UNCERTIFIED`; the overall publication decision remains
`METHOD_ONLY_WEAK_EVIDENCE`.

Verification completed: module compilation, undefined-risk rejection,
patient-cluster bootstrap check, and prediction scope/test exclusion assertions.

Next scientific work: freeze a suffix-based policy family, implement valid
patient-level risk bounds and stronger matched-cost stopping/referral baselines,
then assess independent external data access and confirmatory evidence. The
five-class endpoint, external robustness experiments, and publication manuscript
are still outstanding. No publication-ready result is claimed.

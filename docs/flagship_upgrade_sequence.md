# ActiveRetina-RC Flagship Upgrade Sequence

Date: 2026-10-03

Status: `APPLIED_BEFORE_MODEL_TRAINING`

## Upgrade Summary

The project is upgraded from an MFIDDR-dependent active field selection study into a stronger flagship paper:

**Risk-Controlled Adaptive Retinal Imaging for Diabetic Retinopathy Screening**

The new center is not "which field next?" It is:

> How much retinal evidence is enough to make a safe screening decision, and when must the system acquire more evidence or defer?

This framing is intended to be more journal-grade than the original MFIDDR-only plan because it combines adaptive acquisition with explicit clinical risk control, calibrated deferral, external robustness, and failure analysis.

## Superseded Framing

The following statements are superseded as flagship-level claims:

- MFIDDR is required for the main paper.
- DeepDRiD is only infrastructure de-risking.
- The main contribution is non-myopic four-field planning.
- CCG is the default flagship method.

These remain valid only as a future extension if MFIDDR or another true multi-step same-eye dataset becomes available.

## New Flagship Claim

ActiveRetina-RC learns and calibrates adaptive retinal imaging policies that minimize acquisition burden subject to a missed-referable-DR risk constraint.

Each patient/eye can receive one of three decisions:

- `STOP`: current image evidence is sufficient under the target risk.
- `ACQUIRE_SECOND_VIEW`: another same-eye view is needed.
- `DEFER`: the model cannot safely decide and should route the case to standard imaging or human review.

## Evidence Standard

The paper can advance only from real, patient-separated evidence:

- local dataset integrity audit,
- immutable patient-level train/calibration/validation/test splits,
- cross-fitted predictions for any hindsight or learned value analysis,
- calibration policies selected without sealed-test tuning,
- sealed-test reporting only once per frozen policy family,
- bootstrap confidence intervals for performance and acquisition burden,
- explicit mapping sensitivity for DeepDRiD suffix identity,
- external robustness labeled by valid dataset role.

Fixture outputs, in-sample predictions, and hand-entered metrics cannot support claims.

## Flagship Experiments

1. DeepDRiD paired-view value:
   - first-view only,
   - second-view only,
   - two-view set classifier,
   - both suffix mappings.

2. Adaptive acquisition:
   - always one view,
   - always two views,
   - threshold-based stop/acquire,
   - calibrated stop/acquire/defer,
   - hindsight upper bound from cross-fitted predictions only.

3. Risk control:
   - patient-level calibration,
   - missed-referable-DR risk target,
   - image savings at certified operating points,
   - no certified policy when calibration evidence is insufficient.

4. Robustness:
   - external single-image DR datasets for calibration shift and defer behavior,
   - lesion-quality datasets for failure analysis where available,
   - no promotion of single-image datasets into acquisition evidence.

5. Reporting:
   - cost-risk Pareto curves,
   - defer-rate curves,
   - calibration plots,
   - grade-stratified errors,
   - quality-stratified errors,
   - mapping sensitivity table,
   - external robustness table,
   - explicit limitations.

## Decision Labels

- `FLAGSHIP_GO`: certified adaptive policy saves images on sealed DeepDRiD patients while preserving target missed-referable-DR risk, and external robustness analysis supports the safety framing.
- `METHOD_ONLY_WEAK_EVIDENCE`: the method is sound, but savings, calibration sample size, or robustness are too weak for a flagship empirical claim.
- `NO_GO`: paired-view value or risk-controlled stopping fails.

## Manuscript Boundary

The final paper must not claim:

- MFIDDR results before access,
- four-field planning without a true four-field dataset,
- non-myopic superiority without multi-step evidence,
- official DeepDRiD suffix mapping unless independently confirmed,
- clinical deployment readiness.

The paper may claim:

- a risk-controlled adaptive imaging framework,
- paired-view DeepDRiD acquisition-depth evidence,
- calibrated stop/acquire/defer policies,
- sealed-test operating curves,
- external robustness and failure analysis,
- MFIDDR as a future four-field extension.

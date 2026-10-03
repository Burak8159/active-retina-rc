# ActiveRetina-RC Flagship Execution Plan

Date: 2026-10-03

Flagship status: `UPGRADED — MFIDDR-INDEPENDENT`

The flagship paper has been upgraded from MFIDDR-dependent four-field planning to risk-controlled adaptive retinal imaging. MFIDDR remains valuable, but it is no longer required for the main paper.

## Flagship Claim

ActiveRetina-RC minimizes retinal image acquisition burden subject to a user-specified missed-referable-DR risk constraint. After an initial image, the policy chooses:

- `STOP`: current retinal evidence is sufficient under the calibrated risk target.
- `ACQUIRE_SECOND_VIEW`: more retinal evidence is needed.
- `DEFER`: the model cannot safely decide under the target risk level.

This is stronger than the original MFIDDR-only plan because the main contribution is clinical safety under uncertainty, not only active feature acquisition over retinal fields.

## Main Track: DeepDRiD Risk-Controlled Adaptive Acquisition

DeepDRiD task:

`first available same-eye fundus view -> STOP, acquire paired second view, or DEFER`

Because the public DeepDRiD metadata reviewed so far does not officially map filename suffixes to view identity, all acquisition experiments must be run under both mappings:

- visual-audit mapping: `suffix_1 = optic_disc`, `suffix_2 = macula`
- swapped sensitivity mapping: `suffix_1 = macula`, `suffix_2 = optic_disc`

If conclusions change materially under the swapped mapping, suffix identity remains a blocker for field-named claims. The paper may still report suffix-agnostic paired-view adaptive imaging if the acquisition conclusions are stable.

Execution order:

1. Confirm schema, labels, image integrity, patient splits, and mapping sensitivity branches.
2. Generate cross-fitted predictions for single-view and two-view inputs.
3. Quantify paired-view value with bootstrap confidence intervals.
4. Build fixed policy families for `STOP`, `ACQUIRE_SECOND_VIEW`, and `DEFER`.
5. Calibrate policies on held-out calibration patients using missed-referable-DR risk.
6. Select the lowest-burden certified policy at each target risk level.
7. Evaluate only once on sealed test patients.
8. Stress-test robustness on public external datasets according to their valid evidence roles.
9. Produce cost-risk curves, defer analysis, calibration plots, and failure-mode tables.

## External Robustness Track

Goal: make the paper journal-grade by showing where the risk-controlled framework is robust, fragile, or must defer under shift.

Use public datasets only according to their structure:

- paired same-eye datasets: acquisition-depth evidence when available and legally accessible.
- single-image graded DR datasets: external validation, calibration shift, encoder robustness, and defer behavior.
- lesion datasets: failure analysis and evidence that decisions align with disease features.
- multi-disease datasets: specificity and out-of-distribution stress checks.

Do not present single-image datasets as same-eye acquisition evidence.

## MFIDDR Extension Track

Goal: if MFIDDR arrives, use it to extend the paper or create a follow-up four-field study.

Required checks before any MFIDDR claim:

- official access/download route
- license and research-use conditions
- full training images
- full test images
- four-view labels/metadata
- patient identifiers or reliable grouping
- field identities
- DR labels
- image integrity and missing-view audit

When available, immediately run:

1. schema/integrity audit
2. patient-level split verification/freeze
3. all relevant 1/2/3/4-field subset evaluations
4. multi-view value
5. individualization gap
6. greedy-vs-non-myopic planning gap
7. `GO-CCG`, `GO-MYOPIC`, or true empirical `NO-GO`

## Dataset Expansion Rule

Do not chase datasets as decoration. Each dataset must answer one of these questions:

1. Does acquisition of another same-eye view improve prediction?
2. Does calibration remain reliable under population or imaging shift?
3. Does the policy defer more often when the domain is unsafe?
4. Are errors concentrated in clinically interpretable grades, qualities, or lesion patterns?
5. Does an external dataset reproduce a clinically relevant operating point?

## Locked Work

Do not implement CCG unless MFIDDR or another true multi-step same-eye acquisition dataset shows:

1. real multi-view value
2. real patient-specific individualization
3. real non-myopic planning value

Do not build custom ViTs, hardware, goggles, Bluetooth/phone systems, or deployment layers during this phase. Lesion analysis is allowed only as failure analysis or robustness evidence, not as a new paper center.

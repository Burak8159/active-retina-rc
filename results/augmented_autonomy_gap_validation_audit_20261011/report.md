# Augmented Autonomy Gap Analysis

Validation-selected development candidate: `ActiveRetina-DDR-augmented-selective-fusion-v2`.
The test split has been inspected in prior searches and is diagnostic only, not independent confirmation.

## Validation Split

- n: `1185`
- positive: `759`
- negative: `426`
- clear_positive: `0`
- clear_negative: `185`
- refer_positive: `661`
- refer_negative: `20`
- abstain_positive: `98`
- abstain_negative: `221`
- diagnostic_decisions: `866`
- abstentions: `319`
- diagnostic_coverage: `73.1%`
- abstention_rate: `26.9%`
- negative_clearance_rate: `43.4%`
- referable_referral_rate: `87.1%`
- false_negative_clearances: `0`
- false_positive_referrals: `20`
- specificity_among_diagnostic_decisions: `90.2%`

## Previously Inspected Test Split (Diagnostic Only)

- n: `1435`
- positive: `946`
- negative: `489`
- clear_positive: `0`
- clear_negative: `166`
- refer_positive: `842`
- refer_negative: `34`
- abstain_positive: `104`
- abstain_negative: `289`
- diagnostic_decisions: `1042`
- abstentions: `393`
- diagnostic_coverage: `72.6%`
- abstention_rate: `27.4%`
- negative_clearance_rate: `33.9%`
- referable_referral_rate: `89.0%`
- false_negative_clearances: `0`
- false_positive_referrals: `34`
- specificity_among_diagnostic_decisions: `83.0%`

## What Still Blocks Every-Image Binary Autonomy

- The candidate abstains on `393` of `1435` test images.
- Abstentions include `104` referable and `289` nonreferable images.
- The policy falsely refers `34` nonreferable images among automatic decisions.
- Therefore, the next model improvement target is not lower missed-disease risk inside the accepted selective mode; it is separating abstained/false-positive nonreferable images from hard referable images without introducing any false-negative clearances.

Case lists:

- `test_abstentions.csv`
- `test_false_positive_referrals.csv`

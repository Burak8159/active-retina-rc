# Augmented Autonomy Gap Analysis

Accepted policy: `ActiveRetina-DDR-augmented-selective-fusion-v2`.

## Validation Split

- n: `1185`
- positive: `759`
- negative: `426`
- clear_positive: `0`
- clear_negative: `154`
- refer_positive: `620`
- refer_negative: `17`
- abstain_positive: `139`
- abstain_negative: `255`
- diagnostic_decisions: `791`
- abstentions: `394`
- diagnostic_coverage: `66.8%`
- abstention_rate: `33.2%`
- negative_clearance_rate: `36.2%`
- referable_referral_rate: `81.7%`
- false_negative_clearances: `0`
- false_positive_referrals: `17`
- specificity_among_diagnostic_decisions: `90.1%`

## Held-Out Test Split

- n: `1435`
- positive: `946`
- negative: `489`
- clear_positive: `0`
- clear_negative: `187`
- refer_positive: `752`
- refer_negative: `19`
- abstain_positive: `194`
- abstain_negative: `283`
- diagnostic_decisions: `958`
- abstentions: `477`
- diagnostic_coverage: `66.8%`
- abstention_rate: `33.2%`
- negative_clearance_rate: `38.2%`
- referable_referral_rate: `79.5%`
- false_negative_clearances: `0`
- false_positive_referrals: `19`
- specificity_among_diagnostic_decisions: `90.8%`

## What Still Blocks Every-Image Binary Autonomy

- The accepted policy abstains on `477` of `1435` held-out test images.
- Abstentions include `194` referable and `283` nonreferable images.
- The policy falsely refers `19` nonreferable images among automatic decisions.
- Therefore, the next model improvement target is not lower missed-disease risk inside the accepted selective mode; it is separating abstained/false-positive nonreferable images from hard referable images without introducing any false-negative clearances.

Case lists:

- `test_abstentions.csv`
- `test_false_positive_referrals.csv`

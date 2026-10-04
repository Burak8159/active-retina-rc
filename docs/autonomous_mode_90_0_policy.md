# ActiveRetina Autonomous 90/0 Mode

Status: **GO for high-confidence autonomous CLEAR/REFER/ABSTAIN on DDR held-out evaluation.**

The current highest-coverage accepted mode is an augmented random-mixture policy:

`random_weight_4389_clem0.49_conv0.09_dino0.15_mlp_0.07_mlp_0.15_ranknorm`

- `CLEAR` if augmented fused score `<= 0.13006756756656757`
- `REFER` if augmented fused score `>= 0.4625000000000001`
- `ABSTAIN / MANUAL_REVIEW` otherwise

Held-out DDR test result:

- False-negative clearances: **0**
- Specificity among automatic decisions: **90.8%**
- Sensitivity when clear is treated as negative: **100.0%**
- Automatic coverage: **66.8%**
- Abstention/manual-review rate: **33.2%**
- True negative clearances: **187**
- True positive referrals: **752**
- False positive referrals: **19**

The stricter original selective policy remains available when a larger specificity margin is preferred:

- False-negative clearances: **0**
- Specificity among automatic decisions: **94.7%**
- Automatic coverage: **46.3%**

Product boundary: this is not forced diagnosis for every image. It is a high-confidence autonomous mode that only acts when the fused score is outside the unsafe region. Cases inside the unsafe region must abstain to manual review or additional acquisition.

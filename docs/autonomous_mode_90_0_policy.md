# ActiveRetina Autonomous 90/0 Mode

Status: **GO for high-confidence autonomous CLEAR/REFER/ABSTAIN on DDR held-out evaluation.**

The selected mode is `median_all_ranknorm` with:

- `CLEAR` if fused score `<= 0.05000000000000001`
- `REFER` if fused score `>= 0.5875`
- `ABSTAIN / MANUAL_REVIEW` otherwise

Held-out DDR test result:

- False-negative clearances: **0**
- Specificity among automatic decisions: **94.7%**
- Sensitivity when clear is treated as negative: **100.0%**
- Automatic coverage: **46.3%**
- Abstention/manual-review rate: **53.7%**
- True negative clearances: **72**
- True positive referrals: **588**
- False positive referrals: **4**

Product boundary: this is not forced diagnosis for every image. It is a high-confidence autonomous mode that only acts when the fused score is outside the unsafe region. Cases inside the unsafe region must abstain to manual review or additional acquisition.

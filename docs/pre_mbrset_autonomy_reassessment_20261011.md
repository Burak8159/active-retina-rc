# ActiveRetina Pre-mBRSET Autonomy Reassessment

Date: 2026-10-11  
Status: `NO_GO_80_SPECIFICITY_ZERO_MISS_FULL_BINARY`

## The actual acceptance target

For this project, `80/0/100` means at least 80% specificity on gradable nonreferable eyes (DR grades 0-1), zero referable cases cleared as nonreferable, and a binary `CLEAR` or `REFER` output for every evaluated case. `REACQUIRE` or `INSUFFICIENT_EVIDENCE` is valuable product behavior, but it is not a completed binary diagnosis and cannot be counted as diagnostic coverage.

All local cohorts and partitions have already informed model selection. Results below are development diagnostics, not an untouched certificate or clinical-use evidence.

## What the completed runs actually say

| Evidence | Observed result | Decision |
|---|---|---|
| Mixed-view ConvNeXt baseline | 30.24% validation specificity at zero misses; its repeatedly inspected development test had 3 misses | Best earlier binary baseline; far below 80% |
| Boundary-focused and 320-pixel fine-tuning | 23.65% and 23.05% validation specificity at zero misses | Both worse than baseline; stop these variants |
| Three-source balanced fine-tuning | 24.61% validation specificity at zero misses | Equal source exposure did not solve cross-dataset ranking |
| B5 with a newly initialized risk head | 2.79% paired-source validation specificity at zero misses, 98.6% referral rate | Failure was largely attributable to throwing away the pretrained risk head |
| Validation-only augmented score mixture | 83.0% specificity among the 72.6% of cases given a diagnosis; 0 observed test misses; 27.4% receive insufficient-evidence/reacquire | Useful three-output frontier, but not binary 80/0/100; split is exposed |
| Same augmented mixture with every deferred case forced to `REFER` | 166/489 negatives cleared: 33.9% specificity, 0 observed test misses | Honest binary translation; still far below 80% |
| Full-binary score-mixture search, validation-selected diagnostic candidate | 33.1% test specificity and 1 missed positive on the exposed test | No-go for binary target |
| 80%-specificity validation-selected ensemble triage | 81.5% validation specificity among 78.6% diagnosed cases, zero validation misses; 21.4% insufficient-evidence | Best current 80-target development operating point, but not binary 80/0/100 |
| Same 80%-target policy, exposed-test diagnostic | 72.3% specificity among 78.6% diagnosed cases, zero observed misses among diagnosed cases; 21.4% insufficient-evidence | Does not reproduce the validation specificity; diagnostic only |
| Same policy with insufficient-evidence forced to `REFER` | 44.6% validation binary specificity; 35.8% exposed-test binary specificity; zero observed misses on both splits | 100% binary action coverage, but fails the 80% specificity target |

The former headline of 90.8% specificity at 66.8% coverage was selected from 5,000 random mixtures by ranking their **test** metrics. That makes the test readout selection-biased. The search code has been corrected to rank candidates on validation only, and the result is now labeled `DEVELOPMENT_ONLY_TEST_EXPOSED`. Do not cite that former headline as a clean held-out success.

The 80%-target frontier above was recomputed by selecting the model mixture and low/high decision thresholds on validation only. The selected mixture was `random_weight_3083_clem0.63_conv0.07_mlp_0.09_mlp_0.18`, with validation thresholds `low=0.38412` and `high=0.57614`. The exposed-test figures are a stability diagnostic, not confirmation. The ensemble's `insufficient evidence` action is a genuine third outcome; routing it to `REFER` gives every image a binary action but does not preserve 80% specificity. These are image-level ensemble results and are not interchangeable with the eye-level paired-view model metrics.

On the current paired-source validation set, the B5 random-head run cleared only 5/179 gradable negatives. Reaching 80% there requires clearing at least 144/179 while missing no referable eye. This is a large ranking improvement, not a threshold adjustment.

## Diagnoses from the repeated failures

1. **The decision-boundary ranking is the main model problem.** Grade-1 eyes are assigned high risk alongside grade-2 referable eyes. Lowering the threshold alone trades specificity for missed cases.
2. **Sampler changes have not helped.** Boundary-focused sampling, source-balanced sampling, and larger images each reduced zero-miss validation specificity relative to the previous baseline. Do not repeat these families without a new signal.
3. **A pretrained disease head must be preserved.** The first B5 attempt removed its trained scalar classifier and initialized a replacement risk head randomly. The residual candidate now starts from the pretrained risk logit and learns a zero-initialized correction.
4. **Paired views and domain shift matter.** DRTiD and DeepDRiD are grouped at eye level with patient-disjoint splits. Their view order is treated symmetrically; no unverified suffix-to-anatomy assumption is needed.
5. **A three-output success is not binary autonomy.** Mapping uncertainty to `REFER` preserves sensitivity but sends many true negatives to referral. Mapping it to `CLEAR` risks false-negative clearance. Neither choice makes the old result 80/0/100.
6. **Evidence integrity needs a hard gate.** Repeated searches on the same test split exhausted its confirmatory value. Candidate ranking is now validation-only; test values are labeled diagnostic, and no current cohort can provide final certification.

## Applied changes and current experiment state

- Fixed final threshold selection in `train_mixed_view_bag_autonomy.py` so both validation thresholding and test reporting use the same declared evaluation-source filter. This prevents DDR from affecting a paired-source policy.
- Added a regression test where including DDR would change the paired-source threshold.
- Fixed both score-mixture search utilities to rank candidates on validation rather than test, and to mark prior-inspected test metrics as diagnostic only.
- Added regression tests ensuring test performance cannot select the winning score mixture.
- Ran retinal-pretrained EfficientNet-B5 residual tuning and paired-source experiments. The all-source branch was stopped after epoch 2 because zero-miss specificity remained at 0-2.8% on paired validation. A separate paired-only retry also failed its epoch-2 gate at 2.8% specificity (5/179 negatives cleared, zero positive misses). Neither candidate is promoted.
- A subsequent paired-only launch was active on RTX when SSH became unavailable. Its final state and checkpoint could not be verified; do not assume it completed or use its output. Before another run, reconnect and inspect the exact process/output state to avoid duplicate jobs.
- Recomputed the ensemble frontier for the user's 80% target using validation-only candidate and threshold selection. This improves the selective development operating point, while explicitly failing the forced-binary specificity requirement.

## What can realistically be achieved before mBRSET

The current results do not justify more B5 epochs or another sampler variation: the paired validation zero-miss specificity stayed below 3% in the retry, versus 30.2% for the older mixed ConvNeXt run, and the B5 route is not improving this boundary. Keep the validation-selected ensemble as the current best *selective* research candidate (81.5% conditional specificity at 78.6% validation diagnostic coverage) and route its uncertainty cases to explicit reacquisition/clinician review. Do not describe it as binary 80/0/100 or as autonomous diagnosis for every image. The next model-development intervention should be chosen from an audited list of grade-1 false referrals and grade-2 false clearances from the training split only, and must be a materially different pretrained representation or feature-learning objective; if the resulting validation frontier does not improve, stop that family rather than extend training blindly.

mBRSET can add an independent domain and paired-view confirmation opportunity after access, but it cannot guarantee the target. If the locked model misses on mBRSET, the correct result is a failed confirmation, followed by retraining on permitted development data and a new untouched cohort; never tune against the certification partition and reuse it as evidence.

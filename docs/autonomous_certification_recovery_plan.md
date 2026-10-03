# ActiveRetina-RC Autonomous Certification Recovery Plan

Status: `RECOVERY_PLAN_WITH_DEVELOPMENT_MODEL_UPGRADE_RESULTS`

The clinician-reviewed product is usable as a supervised acquisition assistant.
The autonomous diagnostic claim is blocked by two separate problems:

1. Evidence size: the current online cohort has 48 positive patients, while a
   zero-miss 95%/5% certificate needs 59.
2. Model performance: the locked artifact misses 7 positive patients on the
   online cohort.

Both must be fixed. More data alone is not enough if the model continues to miss
referable patients. Better thresholds alone are not enough if the certification
cohort remains below 59 positive patients.

## Development Harness Result

The model-upgrade harness now trains real candidate heads over the frozen
ResNet-50 paired-view embeddings and evaluates them on the already inspected
DeepDRiD online cohort. This is development evidence only, not a prospective
certificate.

Best development candidate:

- Candidate: `logistic_c025_balanced`
- Online cohort AUROC: `0.9419`
- Average precision: `0.9000`
- Default 0.5 threshold: 11 missed positive patients among 48 decided positive
  patients.
- Best zero-miss mode with at least 50% decided-negative specificity: 48/48
  positive patients decided with zero misses.
- Specificity in that zero-miss mode: `67.3%` among decided negative patients.
- 95% one-sided missed-risk upper bound: `6.05%`.
- Certification status: `NO_GO`, because 48 zero-miss positive patients remain
  below the 59-patient minimum for a 95%/5% certificate.

Interpretation:

The old locked artifact is not the right autonomous mode, but the embedding
space is not dead. The immediate ML task is to freeze the best candidate
operating mode, then run it once on an untouched cohort large enough to make the
certificate possible.

## Route A: Fastest Honest Certification Route

Goal:

`GO_AUTONOMOUS_NARROW_MODE`

Claim:

The system may autonomously identify only a narrow, high-confidence class of
cases while deferring everything else to clinician review.

Steps:

1. Use development data only to freeze a narrow autonomous mode.
2. Treat the already inspected online cohort as development evidence from now
   on, not as a fresh certificate.
3. Obtain an untouched cohort with at least 80 positive patients, preferably
   120.
4. Run the frozen mode once.
5. Claim autonomy only if missed positive patients are zero and the exact 95%
   upper bound is at or below 5%.

Advantages:

- Most realistic path to an honest certificate.
- Keeps product useful because noncertified cases are still handled by the
  clinician-reviewed workflow.

Disadvantages:

- Autonomy coverage may be low.
- Requires another untouched labeled cohort.

## Route B: Stronger Model Route

Goal:

Reduce the 7 online-cohort misses before the next untouched certification set.

Model upgrades:

- Replace logistic heads with calibrated gradient boosting or shallow MLP heads
  over frozen embeddings.
- Add an ensemble across two encoders instead of a single ResNet-50.
- Use the DeepDRiD online cohort only as development/failure-analysis after it
  has already been inspected.
- Add grade-aware loss or conservative referable thresholding.
- Add image-quality gating before prediction.
- Add test-time augmentation only if deterministic and frozen before
  certification.

Required discipline:

- Once the model is chosen, freeze artifact, thresholds, preprocessing, and API
  schema.
- Do not alter the model after viewing the next certification cohort.

## Route C: Data Expansion Route

Goal:

Reach 80-120 positive patients in an untouched certification cohort.

Potential sources:

- MFIDDR if access arrives: best for same-eye multi-field acquisition.
- DRTiD if access is granted: two-field macula/disc evidence.
- mBRSET if credentialed access is granted: portable camera generalization.
- A new locally collected clinician-labeled set: best if the product is meant
  to run with your hardware.
- Single-image datasets such as EyePACS/APTOS/DDR/IDRiD: useful for diagnostic
  robustness, but not sufficient alone for paired-view acquisition claims.

Minimum viable certification cohort:

- Hard minimum if zero misses: 59 positive patients.
- Practical minimum: 80 positive patients.
- Preferred: 120 positive patients.
- Include nonreferable patients too, so specificity and false-positive burden
  can be reported.

## Route D: Paper Strategy

IEEE-ready paper should not claim full autonomous diagnosis yet.

Strong publishable claim now:

`Risk-controlled clinician-reviewed adaptive retinal imaging station with a
public DeepDRiD benchmark, local analysis API, and quantified certification gap.`

Autonomous extension claim later:

`A prespecified narrow autonomous operating mode certified on an untouched
positive-patient cohort.`

This is stronger than pretending the current model is perfect, because it makes
the paper falsifiable, reproducible, and clinically honest.

## Immediate Next Implementation Tasks

1. Freeze `logistic_c025_balanced` as the next autonomous-candidate artifact,
   including thresholds for the zero-miss, specificity-constrained operating
   mode.
2. Prepare an untouched-certification import tool for the next cohort.
3. Require at least 59 positive patients, and preferably 80-120 positive
   patients, before attempting the autonomous certificate.
4. Run the frozen artifact once on that cohort without threshold repair.
5. Keep the current clinical station/API stable while candidate autonomy remains
   behind a separate artifact version and explicit gate.

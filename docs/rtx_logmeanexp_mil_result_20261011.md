# RTX Log-Mean-Exp Paired-View Experiment

Date: 2026-10-11
Status: `DEVELOPMENT_GAIN; AUTONOMY_TARGET_NOT_MET`

## Hypothesis

The prior model fused view embeddings before the disease-risk head. A masked, temperature-controlled log-mean-exp pool of per-view risk logits may preserve a strong signal present in either retinal field. No view-specific labels are assigned.

## Protocol

- Backbone: `convnext_tiny.fb_in22k_ft_in1k`, warm-started from `active_retina_mixed_view_bag_refine_dev_20261010.pt`.
- Patient-disjoint three-source manifest: DDR, DRTiD paired eyes, and DeepDRiD paired eyes; 4,675 training eyes and 982 validation eyes.
- One epoch, image size 224, log-mean-exp temperature 0.25; validation alone selected the zero-miss operating threshold.
- Test evaluation was explicitly skipped. These results are development evidence, not certification.
- RTX script SHA-256 matched the locally tested source: `be922699f4b7301ccef7029777058c2bc70f3336b5e76ac5ddd3f45e78bae0cd`.
- Checkpoint SHA-256: `b8285f5d20b71a751170afb34e4deAd3cc4f7cea8d4333945ca42d9cf77dd71a`.

## Result

At the validation-selected policy, specificity among gradable nonreferable eyes was **37.31% (144/386)**, with **0/463 referable eyes missed**, 100% case-action coverage, and an 85.34% referral rate. The matched three-source source-balanced baseline was **24.61% (95/386)** at zero validation misses, so the new candidate gained 12.70 percentage points on this development validation split.

The result is source-dependent: DDR specificity was 52.17% (108/207), DeepDRiD 25.00% (13/52), and DRTiD 18.11% (23/127), each with zero validation misses. Grade-1 specificity remained only 13.39% (15/112); this is still the central boundary problem.

## Stopping evidence

A two-epoch warm-start continuation from the epoch-1 checkpoint was attempted. Its first epoch fell to **28.24%** zero-miss validation specificity, below the saved 37.31% candidate. It was stopped immediately. The original epoch-1 checkpoint remains the best artifact; no later epoch or test result is claimed.

## Decision

This is a real but modest development improvement, not `80/0/100`, not autonomous certification, and not clinical validation. Do not extend this fine-tuning branch. Keep the epoch-1 checkpoint for reproducibility, and choose any next experiment based on the remaining grade-1 and source-specific errors rather than repeating the same continuation.

Artifacts are in `results/rtx_logmeanexp_mil_gate1_20261011/`; the checkpoint remains on the RTX at `models/active_retina_logmeanexp_mil_gate1_20261011.pt` and is intentionally not copied into the repository.

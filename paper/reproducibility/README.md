# ActiveRetina-RC replay reproduction package

## Scope

This package exposes two deliberately different audit levels. The clean-clone default checks the internal consistency of tracked aggregate summaries. The optional full replay executes the frozen policy against per-eye component-score rows, but those inputs are not distributed here. Neither mode reproduces image preprocessing, component-model inference, or model training, and neither is independent clinical validation. The reported 70.13% remains a same-cohort development replay.

## Run

Use Python 3.12 and install the single dependency in a clean environment:

```sh
python3.12 -m venv .venv-paper-audit
. .venv-paper-audit/bin/activate
python -m pip install -r paper/reproducibility/requirements.txt
python paper/audit_submission_evidence.py --root . --mode summary --output paper/reproducibility/audit_report.json
```

On Windows, activate the environment with `.venv-paper-audit\\Scripts\\Activate.ps1` instead. A successful clean-clone run emits `PASS_TRACKED_AGGREGATE_INTEGRITY_ONLY_NOT_REPLAY` and hashes the tracked summary files. This verifies only that selected reported counts match the expected values in those summaries; it does not independently reconstruct those counts from observations.

The row-level replay is an optional local operation:

```sh
python paper/audit_submission_evidence.py --root . --mode full-replay
```

It requires the two local evidence files below. If they are absent, the command fails closed. A successful run recomputes the actions and scores from supplied component-score rows and checks them against saved summaries. It is still not raw-image reproduction.

## Required local evidence files

The clean Git checkout supports `--mode summary`. The optional `--mode full-replay` additionally requires:

- `results/deployable_exposed_baseline_7013_20261013_with_components/policy.json`
- `results/deployable_exposed_baseline_7013_20261013_with_components/validation_replay.csv`
- the five tracked `summary.json` files named in `paper/audit_submission_evidence.py`

These files are local, untracked, derived policy/score artifacts. They are excluded pending confirmation of redistribution rights for per-eye scores, labels, and reference vectors. Do not copy them into a public repository unless the applicable dataset terms and any access conditions permit it. The aggregate mode does not silently claim to replace this row-level replay.

## Reproduction boundaries

The replay CSV contains precomputed component scores and the policy contains empirical rank-reference values and thresholds. The script executes the policy against those rows; it cannot regenerate those component scores from images. The model checkpoints, complete preprocessing and inference commands, exact dataset release snapshots, and original cross-validation prediction files are not all included. Consequently, this is a deterministic replay package, not a full computational reproduction package.

The source datasets remain at their official distribution channels. Do not add images, restricted metadata, or derived per-eye outputs to the public repository until the release checklist's rights review is resolved. The repository's root `LICENSE` currently says **All rights reserved**; it does not grant an MIT license or other general reuse permission. That repository license also does not override third-party dataset terms.

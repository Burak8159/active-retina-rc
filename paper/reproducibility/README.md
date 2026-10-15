# ActiveRetina-RC replay reproduction package

## Scope

This package reruns the frozen **score-level** replay and checks the manuscript's saved headline counts. It is not an end-to-end reproduction of image preprocessing, component-model inference, model training, or an independent clinical evaluation. The reported 70.13% is the same-cohort development replay described in both manuscripts.

## Run

Use Python 3.12 and install the single dependency in a clean environment:

```sh
python3.12 -m venv .venv-paper-audit
. .venv-paper-audit/bin/activate
python -m pip install -r paper/reproducibility/requirements.txt
python paper/audit_submission_evidence.py --root . --output paper/reproducibility/audit_report.json
```

On Windows, activate the environment with `.venv-paper-audit\\Scripts\\Activate.ps1` instead. A successful run emits `PASS_SCORE_LEVEL_REPLAY_AND_HEADLINE_COUNTS`, recomputes every action and risk score from the input rows, verifies aggregate source counts and saved development summaries, and records SHA-256 hashes of all inputs.

## Required local evidence files

The clean Git checkout currently does **not** contain all inputs needed by the command. The audit requires:

- `results/deployable_exposed_baseline_7013_20261013_with_components/policy.json`
- `results/deployable_exposed_baseline_7013_20261013_with_components/validation_replay.csv`
- the five tracked `summary.json` files named in `paper/audit_submission_evidence.py`

The first two files are presently local, untracked, derived score/policy artifacts. They are not included in this repository release pending a documented data-rights decision for redistribution of per-eye derived scores, labels, and reference vectors. Obtain them only through authorized project storage and under each source's terms. If these inputs are absent, the audit deliberately stops and names them; it does not silently downgrade to a count-only check.

## Reproduction boundaries

The replay CSV contains precomputed component scores and the policy contains empirical rank-reference values and thresholds. The script executes the policy against those rows; it cannot regenerate those component scores from images. The model checkpoints, complete preprocessing and inference commands, exact dataset release snapshots, and original cross-validation prediction files are not all included. Consequently, this is a deterministic replay package, not a full computational reproduction package.

The source datasets remain at their official distribution channels. Do not add images, restricted metadata, or derived per-eye outputs to the public repository until the release checklist's rights review is resolved. Repository MIT licensing applies to project code only and does not override third-party dataset terms.

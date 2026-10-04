# ActiveRetina-RC Clinical Product Readiness

Status: `GO_CLINICIAN_REVIEWED_ACQUISITION_ASSISTANT`

Additional computational status: `GO_AUGMENTED_THREE_OUTPUT_AUTONOMOUS_TRIAGE`

This package now has a software-side clinical workflow: a locked model artifact,
real image/case input handling, local browser station, JSON output, HTML case
reporting, audit hashes, hardware-enclosure runbook, and fail-closed decision
gating. It is ready as a clinician-reviewed acquisition assistant prototype.
Separately, the DDR augmented selective-autonomy layer now reaches zero
false-negative clearances, 90.8% specificity among automatic decisions, and
66.8% automatic coverage on held-out evaluation. It is still not an unrestricted
autonomous diagnosis system because uncertain cases must return
`AUTONOMOUS_INSUFFICIENT_EVIDENCE_REACQUIRE` rather than receiving a forced
binary diagnosis.

## Completed software-side controls

- Locked model version string.
- Locked trainable artifact: `models/active_retina_rc_clinical_artifact.joblib`.
- Artifact metadata with manifest and feature-cache hashes:
  `models/active_retina_rc_clinical_artifact.metadata.json`.
- Real case runner from retinal image paths:
  `src/active_retina/clinical_case_runner.py`.
- Local browser station:
  `src/active_retina/clinical_station.py`.
- Clinician-readable HTML report:
  `results/clinical_case_demo_report.html`.
- Machine-readable case result:
  `results/clinical_case_demo_output.json`.
- Live encoder case result:
  `results/clinical_case_live_encoder_output.json`.
- Hardware/enclosure station runbook:
  `docs/hardware_enclosure_clinical_station.md`.
- Explicit input/output schema.
- Fail-closed decision gateway.
- Manual-review requirement on every output.
- Clinical-use blocker when `intended_use` is not `research`.
- Anatomy-mapping blocker.
- Image-quality blocker.
- Immutable JSONL audit log.
- Input payload SHA-256 hash.
- Reproducible full DeepDRiD cross-validation evidence.
- Formal calibration risk-bound audit.
- Software smoke tests for the gateway and clinician report.
- Frozen augmented selective-autonomy policy artifact:
  `results/autonomous_mode_augmented_policy_v2.json`.
- Real autonomy frontier plot:
  `results/autonomy_extra_signal_clementp_efficientnet_b5/frontier.svg`.

## Autonomous-use blockers

- No unrestricted every-image binary diagnostic mode. The current accepted
  autonomous layer is three-output: nonreferable, referable, or insufficient
  evidence/reacquire.
- No regulatory clearance.
- No clinical site validation.
- No verified DeepDRiD suffix-to-anatomy mapping for field-named claims.
- No integration with real fundus camera DICOM/worklist systems.
- No HIPAA/GDPR production deployment review.
- No ophthalmologist usability validation.

## Permitted current use

Clinician-reviewed acquisition assistant:

```bash
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.clinical_gateway \
  --input configs/clinical_gateway_example.json \
  --output results/clinical_gateway_example_output.json
```

Run the locked model artifact on a paired DeepDRiD image case:

```bash
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.clinical_case_runner \
  --case-id DEMO-IMG-001 \
  --first-image external/DeepDRiD_v1_1/deepdrdoc-DeepDRiD-56d8af7/regular_fundus_images/regular-fundus-training/Images/1/1_l2.jpg \
  --second-image external/DeepDRiD_v1_1/deepdrdoc-DeepDRiD-56d8af7/regular_fundus_images/regular-fundus-training/Images/1/1_l1.jpg \
  --output-json results/clinical_case_demo_output.json \
  --output-html results/clinical_case_demo_report.html
```

Run live image inference with the project-local encoder cache:

```bash
HF_HOME=.model_cache/huggingface TORCH_HOME=.model_cache/torch \
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.clinical_case_runner \
  --case-id DEMO-LIVE-ENCODER-001 \
  --first-image external/DeepDRiD_v1_1/deepdrdoc-DeepDRiD-56d8af7/regular_fundus_images/regular-fundus-training/Images/1/1_l2.jpg \
  --second-image external/DeepDRiD_v1_1/deepdrdoc-DeepDRiD-56d8af7/regular_fundus_images/regular-fundus-training/Images/1/1_l1.jpg \
  --force-live-encoder \
  --output-json results/clinical_case_live_encoder_output.json \
  --output-html results/clinical_case_live_encoder_report.html
```

Run the local browser station:

```bash
HF_HOME=.model_cache/huggingface TORCH_HOME=.model_cache/torch \
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.clinical_station \
  --host 127.0.0.1 \
  --port 8765
```

Every output is marked:

`manual_review_required: true`

This is intentional and must not be removed until the model passes a prespecified
clinical validation protocol.

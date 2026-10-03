# ActiveRetina-RC Clinical Station Hardware Enclosure

Status: `PRODUCT_GRADEABLE_CLINICIAN_REVIEWED_PROTOTYPE`

This document defines the practical hardware/software station needed to run
ActiveRetina-RC as a clinician-reviewed acquisition assistant. It does not claim
autonomous diagnosis. The station is designed so a real operator can import
fundus photographs, run the locked model, review the recommendation, and store a
case audit trail.

## Intended Physical Setup

- Fundus camera or retinal image source exporting JPEG/PNG images.
- Local workstation or mini PC running the ActiveRetina-RC station.
- Monitor visible to the operator.
- Keyboard/mouse or touchscreen.
- Enclosure that prevents cable strain, accidental unplugging, and patient-side
  contact with the compute unit.
- Local project cache containing the pretrained encoder weights:
  `.model_cache/huggingface`.

## Enclosure Requirements

- Non-patient-contact enclosure for the compute device.
- Ventilated shell with unobstructed air intake/exhaust.
- Front USB-C/USB-A or SD-card import port, depending on camera workflow.
- Rear power entry with strain relief.
- Clearly separated operator side and patient side.
- Physical label: `Clinical decision support prototype - clinician review required`.
- No exposed sharp edges.
- Stable base; no free-hanging cables near patient chair.
- Optional VESA or cart mount.

## Software Station

Run the local browser station:

```bash
HF_HOME=.model_cache/huggingface TORCH_HOME=.model_cache/torch \
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.clinical_station \
  --host 127.0.0.1 \
  --port 8765
```

Open:

`http://127.0.0.1:8765`

The station accepts:

- case ID,
- operator ID,
- first fundus image,
- optional second fundus image,
- operator image-quality gate.

The station writes per-case artifacts under:

`clinical_station_cases/<case_id>/`

Each case contains:

- uploaded first image,
- uploaded second image if provided,
- `result.json`,
- `report.html`.

## Operator Workflow

1. Capture the first fundus image with the camera.
2. Export/import the image to the station.
3. Enter case ID and operator ID.
4. Mark image quality as checked or failed.
5. Run the station.
6. If the system asks for a second view, capture/import the second image.
7. Review the report.
8. Treat all recommendations as clinician-reviewed decision support.
9. Save/export the HTML and JSON case record.

## Product Boundary

Current supported mode:

`GO_CLINICIAN_REVIEWED_ACQUISITION_ASSISTANT`

Current blocked mode:

`NO_GO_AUTONOMOUS_DIAGNOSIS`

Reason:

The exact finite-sample audit shows that a 95% confidence, 5% missed-referable-DR
certificate would require at least 59 automatically decided positive-patient
calibration cases with zero misses. The current best zero-miss calibration
candidate has 21 decided positive-patient cases and a 95% upper bound of 13.3%.

## Human-Use Checklist

Before using the station on a real human in a supervised, nonautonomous setting:

- Verify the fundus camera produces readable JPEG/PNG exports.
- Verify imported files open correctly on the station.
- Run one known sample case through the browser station.
- Confirm `manual_review_required: true` in the output.
- Confirm `autonomous_diagnosis: false` in the output.
- Confirm the report opens from `/case?id=<case_id>`.
- Confirm the operator can distinguish first and second views.
- Keep the model output out of the patient-facing screen unless a clinician is
  present to interpret it.

## Failure Modes

- If image quality is failed or uncertain, the station defers to clinician.
- If only one view is supplied and the score is uncertain, the station asks for
  the second view.
- If the encoder cache is missing, live inference may fail until the pretrained
  weights are cached under `.model_cache`.
- If the case ID already exists, files may be overwritten; use unique case IDs
  for real sessions.


# RFMiD Multi-Disease Local Station

This separate local station uses the ConvNeXt-Tiny checkpoint from the RFMiD
development run. It does not replace the existing DR-only clinical-review
station and does not imply clinical validation.

## Start

The checkpoint is a local, ignored artifact and must exist at
`models/rfmid_convnext_tiny_multilabel_20261021.pt`.

```bash
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.rfmid_multidisease_station
```

Open `http://127.0.0.1:8767`. The server binds to loopback only; external/LAN
access is intentionally disabled because this research interface has no user
authentication, TLS termination, or production security controls.

## Browser workflow

Upload one JPEG, PNG, or WebP fundus image (maximum 16 MB). The browser displays
the RFMiD `Disease_Risk` output separately from each condition output, with its
measured test sensitivity, specificity, accuracy, and false-negative count. A score
at or above 0.5 is labeled `FLAGGED`; otherwise it is `NOT_FLAGGED`. These are
independent model outputs, not a mutually exclusive diagnosis and not a claim
that the condition is absent. Reference metrics shown beside each condition
are the measured RFMiD test-split metrics from the accompanying run report.

Images are decoded and processed in memory and are not written to disk by this
station. Do not submit identifiable or clinical images to the research server.

## JSON API

`GET /api/health` reports whether the model loaded.

`POST /api/predict` accepts `multipart/form-data` with one file field named
`image`. Example:

```bash
curl -F 'image=@fundus.jpg' http://127.0.0.1:8767/api/predict
```

The response has one `dataset_overall_risk_label` object and a `conditions`
array with an independent score, fixed threshold, flag, and RFMiD test reference
for every condition head. It includes `image_saved: false` and
`autonomous_diagnosis: false`. Uploads larger than 16 MB and non-image files are
rejected. No case records or uploaded image content are persisted.

This software is research-only. The model has false negatives, no verified
patient-disjoint test split, and no clinical calibration. Outputs must not be
used to diagnose, clear, refer, or otherwise manage a patient.

# ActiveRetina-RC Analysis API

Status: `VERIFIED_LOCAL_API`

The clinical station exposes a local JSON API for integration with a hardware
sequence, camera-import process, or external operator UI.

The browser station is intentionally simpler than the research workspace:
upload a retinal image, select whether it is visibly clear, and receive one
plain-language assessment plus the next step. It does not display development
baseline names or cohort metrics. The result remains a clinician-reviewed
model assessment, not an autonomous diagnosis.

## Endpoint

`POST /api/analyze`

Content type:

`application/json`

## Request

```json
{
  "case_id": "API-DEMO-001",
  "operator_id": "hardware-sequence",
  "first_image_path": "path/to/first.jpg",
  "second_image_path": "path/to/second.jpg",
  "quality_ok": true
}
```

`second_image_path` may be omitted. If the first-view score is uncertain and no
second image is supplied, the response asks for another retinal view.

## Response

The API returns a concise user-facing result:

- case ID,
- plain-language assessment,
- next step,
- clinician-confirmation flag,
- autonomous-diagnosis flag,
- report URL.

Full scores, model metadata, and image hashes remain in the locally saved case
record and detailed report rather than the scan response.

Example verified response:

```json
{
  "case_id": "API-DEMO-001",
  "assessment": "Lower likelihood of referable diabetic retinopathy",
  "next_step": "A clinician must confirm this assessment.",
  "clinician_confirmation_required": true,
  "autonomous_diagnosis": false,
  "report_url": "/case?id=API-DEMO-001"
}
```

## Verified Command

```bash
curl -s -X POST http://127.0.0.1:8765/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"case_id":"API-DEMO-001","operator_id":"hardware-sequence","first_image_path":"external/DeepDRiD_v1_1/deepdrdoc-DeepDRiD-56d8af7/regular_fundus_images/regular-fundus-training/Images/1/1_l2.jpg","second_image_path":"external/DeepDRiD_v1_1/deepdrdoc-DeepDRiD-56d8af7/regular_fundus_images/regular-fundus-training/Images/1/1_l1.jpg","quality_ok":true}'
```

The saved verified response is:

`results/api_demo_response.json`


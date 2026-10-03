# DRCR Protocol AA Public Dataset Audit

Status: `PUBLIC_TABLES_AVAILABLE_IMAGES_NOT_INCLUDED`

Source:

`https://public.jaeb.org/drcrnet/stdy/239`

Local files:

- `/Users/burakaksu/Downloads/Protocol_AA.zip`
- `external/DRCR_AA_public/unpacked/`

## What Was Downloaded

The public package contains protocol documents, case report forms, and tabular
data exports. It does not contain retinal image files.

Important tables:

- `AADRSSData.txt`
- `AAJoslinRCGradingResult.txt`
- `AAPhoto.txt`
- `AAUWFPhoto.txt`
- `ImageLogIn.txt`
- `PtRoster.txt`

## Schema Findings

The package contains rich reading-center grading metadata:

- 388 patients.
- 14,508 image-log rows.
- 11,628 DR severity rows in `AADRSSData.txt`.
- 1,126,265 reading-center grading-result rows in
  `AAJoslinRCGradingResult.txt`.
- ETDRS standard grading templates are present.
- UWF masked/unmasked grading templates are present.
- ETDRS fields 1--7 are represented in the grading table.
- UWF peripheral zones/fields are represented in the grading table.
- Baseline photo table records conventional 7-field photography.
- UWF photo table records on-axis and steered UWF acquisitions.

## ActiveRetina Use

This dataset is scientifically valuable for:

- Formal paper context around ETDRS 7-field versus UWF DR severity grading.
- Designing a future multi-field objective beyond two-view DeepDRiD/DRTiD.
- Citing real clinical-study evidence that field coverage and peripheral
  lesions matter.
- Building non-image simulations or decision-theory appendices from published
  grading tables.

It cannot currently be used for:

- Running the ActiveRetina image model.
- Certifying the autonomous diagnosis artifact.
- Extracting ResNet embeddings.
- Creating a same-eye image manifest.
- Replacing mBRSET/DRTiD/MFIDDR as an image-level certification cohort.

## Decision

`NO_GO_AS_IMAGE_CERTIFICATION_COHORT`

`GO_AS_TABLE_LEVEL_CLINICAL_GRADING_EVIDENCE`

This is a strong paper-supporting dataset, not a final ML-product dataset.

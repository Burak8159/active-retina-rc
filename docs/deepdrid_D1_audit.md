# DeepDRiD D1 Audit

Status: `SCHEMA_AND_LOCAL_IMAGES_VERIFIED`

DeepDRiD is the primary available dataset for the upgraded ActiveRetina-RC flagship. It can support two-view adaptive-depth, calibrated stopping, and defer analyses. It is not a substitute for MFIDDR for four-field field-selection or non-myopic multi-step planning claims.

## Verified From Official Source

Official source: `https://github.com/deepdrdoc/DeepDRiD`

License file: Creative Commons Attribution-ShareAlike 4.0 International.

Regular fundus structure:

- `regular-fundus-training`: 1200 regular fundus images from 300 patients.
- `regular-fundus-validation`: 400 regular fundus images from 100 patients.
- `Online-Challenge1&2-Evaluation`: 400 regular fundus images from 100 patients, with label files now present in the repository.
- each patient has left/right and suffix `1`/`2` images in the public metadata.

Label columns:

- `patient_id`
- `image_id`
- `image_path`
- `Overall quality`
- `left_eye_DR_Level`
- `right_eye_DR_Level`
- `patient_DR_Level`
- `Clarity`
- `Field definition`
- `Artifact`

DR grade mapping:

- `0`: no apparent retinopathy
- `1`: mild NPDR
- `2`: moderate NPDR
- `3`: severe NPDR
- `4`: PDR
- `5`: image quality too low to diagnose/grade

ActiveRetina referable mapping:

- referable DR = grade `>= 2` and `<= 4`
- grade `5` is not a valid disease grade and must be handled as ungradable/quality failure, not as referable DR.

## Field Mapping

Official DeepDRiD sources confirm that regular fundus images are dual-view per eye:

- macula-centered
- optic-disc-centered

Public metadata reviewed so far does not explicitly state the suffix mapping for `1`/`2`.
The current ActiveRetina-RC pipeline therefore uses this documented visual-audit mapping:

- suffix `1` -> `optic_disc`
- suffix `2` -> `macula`

Mapping status: `INFERRED_NOT_OFFICIAL_VISUAL_AUDIT`.

Evidence basis:

- sample images `1_l1`, `1_r1` place the optic disc closer to the image center than their paired suffix-`2` images.
- sample images `1_l2`, `1_r2` place the macula/foveal region closer to the image center.
- the official paper/repository confirm the two fields exist, but not the public filename suffix convention.
- the official archive `Readme.docx` files for training, validation, and online evaluation were extracted and reviewed; they document image IDs, quality labels, and DR labels, but do not define suffix `1`/`2` as macula or optic-disc centered.

Use this mapping for the visual-audit branch and run the swapped branch as mandatory sensitivity analysis. In manuscript claims, describe field identity as visually inferred unless an official suffix statement is found later.

## Current Blockers

- Cross-fitted predictions have not yet been generated.
- Risk-controlled stop/acquire/defer policies have not yet been calibrated.
- The suffix-to-field mapping is sufficient for paired-view acquisition experiments, but remains a documented provenance risk for field-named claims until official confirmation is found.

## Upstream Inventory

Saved to `results/deepdrid_upstream_image_inventory.json`.

- training images: 1200 JPGs across 300 patient directories, four JPGs per patient directory.
- validation images: 400 JPGs across 100 patient directories, four JPGs per patient directory.
- online evaluation images: 400 JPGs across 100 patient directories, four JPGs per patient directory.

Current local image sample:

- `external/DeepDRiD_metadata/sample_images/1_l1.jpg`
- `external/DeepDRiD_metadata/sample_images/1_l2.jpg`
- `external/DeepDRiD_metadata/sample_images/1_r1.jpg`
- `external/DeepDRiD_metadata/sample_images/1_r2.jpg`

## Outputs

- `configs/deepdrid.yaml`
- `results/deepdrid_manifest.csv`
- `results/deepdrid_multiview_manifest.jsonl`
- `results/deepdrid_D1_audit.json`
- `results/deepdrid_splits_manifest.json`
- `results/deepdrid_upstream_image_inventory.json`

## Next Step

Generate cross-fitted single-view and two-view predictions for both mapping branches. Keep suffix mapping provenance visible in reports and update it if official confirmation contradicts the visual audit.

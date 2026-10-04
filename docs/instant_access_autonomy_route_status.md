# Instant-Access Autonomy Route Status

Updated: 2026-10-04

## Current decision

`ROUTE_APPLIED_BUT_NO_GO_FULL_BINARY_AUTONOMY_YET`

The route is now executable, but the current end-to-end forced-binary model still does not meet the target. The immediate product direction remains:

`CLEAR / REFER / REACQUIRE`

not forced `CLEAR / REFER` for every single image.

## What was applied

### 1. Public dataset audit

Created:

- `src/active_retina/audit_instant_access_hf_datasets.py`
- `results/instant_access_hf_dataset_audit.json`
- `results/instant_access_hf_dataset_matrix.csv`

Result: all 9 targeted Hugging Face datasets were accessible.

High-priority accessible datasets:

- `usama10/retinal-dr-longitudinal`
- `ramankamran/retina-age-analysis`

Secondary accessible datasets:

- `macular/diabetic-retinopathy-grading-africa`
- `macular/diabetic-retinopathy-screening-benchmark-africa`
- `ctmedtech/EYEPACS`
- `ctmedtech/DDR-dataset`
- `OctoMed/Aptos`
- `vermayuvraj/idrid-vqa`
- `ctmedtech/RFMID`

### 2. Combined autonomy manifest

Created:

- `src/active_retina/build_autonomy_route_manifest.py`
- `results/autonomy_route_combined_manifest_resplit.csv`
- `results/autonomy_route_combined_manifest_resplit_summary.json`

Current combined development manifest:

- 7,338 images
- 5,788 patient/eye units
- 4,049 referable images
- 3,289 nonreferable images
- train/valid/test split: 5,168 / 1,126 / 1,044
- sources: DDR HF subset + DRTiD paired-view data

Boundary: this is a development/training manifest, not a sealed certificate.

### 3. RTX route execution

The RTX machine is connected and usable.

Data sync to RTX was performed for:

- `external/DDR_hf_specificity300`
- `external/DRTiD_download`

The first real full-data run used:

- EfficientNet-B0
- 384 px images
- 8 epoch plan
- target: 80% specificity, zero missed referable DR

Observed validation metrics:

- Epoch 1: zero misses, 0.65% specificity
- Epoch 2: zero misses, 2.39% specificity

Stopped because it was far from target.

The negative-recovery run used:

- EfficientNet-B0
- 224 px images
- stronger grade-0/1 and nonreferable pressure
- lower positive overprotection

Observed validation metrics:

- Epoch 1: zero misses, 0.87% specificity
- Epoch 2: zero misses, 1.74% specificity

Stopped because it was still far from target.

### 4. Quality/anatomy metadata route

Created:

- `src/active_retina/prepare_retina_age_hf.py`
- `results/retina_age_quality_manifest.csv`
- `results/retina_age_quality_summary.json`

Retina-age-analysis metadata:

- 9,857 images
- 5,393 patients
- patient-level train/valid/test split
- 698 DR-positive binary labels
- camera/anatomy metadata available

Important limitation:

- every row has `quality = Adequate`
- the mirror is not useful for training a bad-image reacquisition detector
- it remains useful for external metadata robustness and anatomy/camera stress testing

## What the results mean

The model is not failing because we lack a script or because the RTX path is broken. The current forced-binary objective learns a conservative zero-miss threshold by referring almost every image.

That is a real model limitation:

- zero-miss threshold is reachable
- useful specificity at full forced coverage is not currently reachable
- repeated loss-weight tweaks are not enough

## Next concrete route

The next viable route is not another threshold search.

1. Build `CLEAR / REFER / REACQUIRE` as the product-level autonomous interface.
2. Train a real image-quality/reacquisition gate using a dataset that includes inadequate or ungradable images.
3. Keep DR diagnosis binary only after image quality is acceptable.
4. Add external single-image robustness datasets for diagnosis hardening:
   - Africa grading/screening
   - APTOS
   - EyePACS
   - RFMiD for confounders
5. Freeze model and run a sealed certification cohort.

## Current claim boundary

Supported:

- autonomous safe-clearance/refer pipeline with reacquisition/deferral
- public-data executable route
- reproducible combined manifest
- RTX training harness

Not supported yet:

- forced binary autonomous diagnosis for every image
- 80% or 90% specificity at zero misses and 100% single-image coverage
- IEEE-level final clinical autonomy claim


# Immediate Dataset Deep Search

Status: `NO_IMMEDIATE_FULL_CERTIFICATION_DATASET_FOUND`

Date: `2026-10-03`

## Search Target

The target was a dataset that can finish the autonomous ActiveRetina
certificate immediately:

- Immediate legal access.
- Actual retinal image files.
- Same-eye paired macula-centered and optic-disc-centered views, or richer
  same-eye multi-field views.
- Patient/eye identifiers.
- DR severity labels mappable to referable DR.
- Enough positive patients for finite-sample certification.
- Not already inspected or used by this project.

## Findings

### DRTiD

Decision: `USED_FOR_DEVELOPMENT_NOW_NOT_UNTOUCHED`

DRTiD has the right structure: paired macula and optic-disc views with DR
grades. We already downloaded and used it. The original candidate failed; the
recovery candidate passes the finite-sample arithmetic on inspected DRTiD test
replay, but DRTiD cannot now be used as a final untouched certificate.

### mBRSET

Decision: `BEST_FINAL_ROUTE_NOT_IMMEDIATE`

mBRSET remains the best final route because it is portable-camera retinal
screening data with patient IDs, ICDR labels, and reported macula/optic-disc
view positions. It requires PhysioNet credentialed access and DUA completion,
so it is not immediate.

### MFIDDR

Decision: `BEST_MULTI_FIELD_ROUTE_NOT_IMMEDIATE`

MFIDDR has the best structure for four-field adaptive acquisition. Access
depends on the dataset authors.

### DeepDRiD

Decision: `ALREADY_USED_NOT_UNTOUCHED`

DeepDRiD is useful and already central to the project, but it cannot be reused
as the final untouched certificate.

### DRCR Retina Network Protocol AA

Decision: `GO_TABLE_LEVEL_EVIDENCE_NO_GO_IMAGE_CERTIFICATION`

The public download contains rich ETDRS 7-field and UWF reading-center grading
tables, but no retinal image files. It strengthens the clinical/paper argument
around field coverage and peripheral lesions, but cannot run the model.

### BRSET

Decision: `GOOD_EXTERNAL_SINGLE_IMAGE_ROUTE_NOT_IMMEDIATE_FOR_US`

BRSET has 16,266 fundus images from 8,524 patients, patient IDs, ICDR labels,
and rich metadata, but PhysioNet lists it under credentialed access/DUA. The
dataset is primarily macula-centered/fovea-centered rather than paired
macula/disc acquisition. It is useful for robustness after access, not for
paired-view acquisition certification.

### IBIA OPTHS_1000000049

Decision: `PROMISING_OPEN_ACCESS_BUT_NOT_DR_CERTIFICATION_READY`

The IBIA listing describes 500 retinal fundus images, 230 patients, bilateral
macula-centered and optic-disc-centered views, metadata, and partial lesion
masks. This is structurally interesting for ActiveRetina. However, the public
description emphasizes multi-disease biomarkers, systemic comorbidities, and
partial lesion masks rather than ICDR/ETDRS DR severity grades. It is also much
smaller than the preferred certification size. Terminal DNS access failed during
this audit, so local image verification remains pending.

If downloaded, this dataset should be treated as:

- `GO` for software/importer smoke testing.
- `GO` for lesion/quality robustness exploration if labels are present.
- `NO-GO` for final autonomous referable-DR certification unless metadata
  contains DR severity labels and enough positive patients.

### Harvard-FairVision DR

Decision: `LARGE_ROBUSTNESS_DATASET_NOT_PAIRED_VIEW_CERTIFICATE`

The public repository points to a Google Drive dataset. The DR subset is about
383 GB and includes DR subtype labels, SLO fundus images, OCT B-scans, and
demographic attributes. This can strengthen diagnostic robustness/fairness and
multimodal analysis, but it is SLO/OCT rather than same-eye macula/disc color
fundus paired acquisition. It is not a direct replacement for mBRSET/MFIDDR.

### GDRBench / DDR / IDRiD / APTOS / Messidor / EyePACS / FGADR / RLDR

Decision: `GO_FOR_SINGLE_IMAGE_GENERALIZATION_NO_GO_PAIRED_CERTIFICATION`

GDRBench aggregates multiple public single-image DR grading datasets and is
valuable for domain-generalization training/evaluation. These datasets do not
provide verified same-eye macula/disc acquisition pairs, so they cannot certify
the paired-view acquisition product. They can strengthen the diagnostic head and
paper by testing cross-domain DR grading robustness.

### DR10K

Decision: `STRUCTURALLY_EXCELLENT_BUT_NOT_PUBLICLY_DOWNLOADABLE_FOUND`

DR10K is highly relevant because the paper reports both macula-centered and
optic-disc-centered images per eye. During this search, no immediate public
download route was found. It should be treated as a potential author-request
route, not an immediate dataset.

## Bottom Line

No immediate dataset found satisfies all final autonomous-certification
requirements.

The best immediate actions are:

1. Use GDRBench-style single-image public datasets to improve robustness.
2. Use DRCR Protocol AA as table-level clinical evidence.
3. Try IBIA OPTHS_1000000049 as an image-import smoke/lesion robustness dataset
   if the open-access download can be obtained.
4. Finish PhysioNet credentialing for mBRSET; it remains the shortest true
   final-certification path.

Current final gate:

`NO_IMMEDIATE_FULL_CERTIFICATION_DATASET_FOUND`

`MBRSET_REMAINS_PRIMARY_FINAL_CERTIFICATION_ROUTE`

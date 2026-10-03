# DRTiD Access Action Pack

Status: `PARALLEL_ACCESS_ROUTE_READY`

DRTiD is the best parallel route while PhysioNet mBRSET credentialing is
pending.

## Why DRTiD Helps

The official repository describes DRTiD as a two-field diabetic retinopathy
dataset with:

- 3,100 two-field fundus images
- paired macula-centric and optic-disc-centric images
- DR severity grade labels
- optic/macula localization metadata
- research-only access through an application form

This matches the ActiveRetina paired-view acquisition problem more directly
than single-image datasets.

## Access Route

Official repository:

`https://github.com/FDU-VTS/DRTiD`

Access form:

`https://docs.google.com/forms/d/e/1FAIpQLSfEYIwk5G1Y6sAcDEjkt7qmv5FWcdG9Jn_P-gSnZ77MWvRM3A/viewform?usp=sf_link`

Contact listed in the official repository:

`jlhou18@fudan.edu.cn`

## After Download

Place the dataset as:

```text
external/DRTiD/
  Original Images/
  Ground Truths/
    DR_grade/
      a. DR_grade_Training.csv
      b. DR_grade_Testing.csv
    Optic_Macula_Localization/
      op_ma_localization.csv
```

Run:

```bash
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.run_untouched_certification_package \
  --cohort-kind drtid \
  --drtid-root external/DRTiD \
  --drtid-include-training \
  --drtid-include-testing \
  --attestation configs/untouched_attestation_template.json \
  --output-dir results/drtid_untouched_certification
```

## Certification Boundary

The official code examples use `Macula`, `Optic disc`, and `Grade` columns.
If the downloaded CSVs do not include a true patient identifier, DRTiD can
provide pair-level untouched validation but not a strict patient-level
autonomous certificate. If a true patient ID exists in the downloaded metadata,
the importer will use it automatically.

Therefore:

- With true patient IDs and enough positives: DRTiD may support full
  patient-level certification.
- Without true patient IDs: DRTiD strengthens the IEEE/arXiv paper and can be a
  serious external paired-view validation, but mBRSET remains the better final
  patient-level certification route.


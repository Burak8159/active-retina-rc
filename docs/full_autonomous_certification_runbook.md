# Full Autonomous Certification Runbook

Status: `MODEL_AND_CLINICAL_PROTOCOL_NOT_READY_FOR_CONFIRMATORY_USE`

This is the shortest path from a legally obtained external paired-view cohort to
a final autonomous-computation decision.

## Required Input

Use either:

1. mBRSET-style metadata with columns for patient, file, ICDR grade, laterality,
   and view position; or
2. A canonical paired-eye CSV with:

```text
patient_id,eye_id,macula_image_path,optic_disc_image_path,grade,referable_dr
```

The cohort must be untouched for ActiveRetina:

- not used for training
- not used for threshold selection
- not inspected before freeze
- no patient/eye/image overlap with DeepDRiD development data
- at least 59 referable-DR positive patients, preferably 80-120+

## mBRSET Path

After credentialed PhysioNet access, place mBRSET under a local folder such as:

```text
external/mBRSET/
```

Then run:

```bash
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.run_untouched_certification_package \
  --cohort-kind mbrset \
  --labels external/mBRSET/<metadata-file>.csv \
  --image-root external/mBRSET/<image-folder> \
  --attestation configs/untouched_attestation_template.json \
  --output-dir results/mbrset_untouched_certification
```

If the downloaded metadata matches the public GitHub CSV and lacks an anatomy
column, first verify the filename suffix-to-anatomy mapping by manual image
audit. Then pass the audited map explicitly:

```bash
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.run_untouched_certification_package \
  --cohort-kind mbrset \
  --labels external/mBRSET/dataframe_brsetmobile.csv \
  --image-root external/mBRSET/images \
  --suffix-view-map 1:macula,2:optic_disc,3:macula,4:optic_disc \
  --attestation configs/untouched_attestation_template.json \
  --output-dir results/mbrset_untouched_certification
```

Do not use a suffix map without official confirmation or manual visual audit.

The command will:

1. Create a certification manifest only if same-eye macula/optic-disc pairs are
   identifiable.
2. Validate untouched status and sample size.
3. Run the frozen autonomous candidate exactly once.
4. Write `final_deployability_decision.json`.

## Canonical External Cohort Path

```bash
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.run_untouched_certification_package \
  --cohort-kind canonical \
  --canonical-input external/untouched_paired_cohort.csv \
  --attestation configs/untouched_attestation_template.json \
  --output-dir results/external_untouched_certification
```

## DRTiD Parallel Path

While PhysioNet credentialing is pending, request DRTiD access through the
official repository/form. After download:

```bash
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.run_untouched_certification_package \
  --cohort-kind drtid \
  --drtid-root external/DRTiD \
  --drtid-include-training \
  --drtid-include-testing \
  --attestation configs/untouched_attestation_template.json \
  --output-dir results/drtid_untouched_certification
```

Important boundary: if the downloaded DRTiD grade CSVs contain only
`Macula`, `Optic disc`, and `Grade`, the run is pair-level external validation.
If the metadata also contains a true patient identifier, the importer uses it
and the run can support patient-level certification.

## Decision Rule

This status means only that the locked retrospective `80/0/100` computational
target was met on an untouched cohort. It does not establish clinical
validation or authorize autonomous clinical diagnosis:

```text
GO_RETROSPECTIVE_BINARY_80_0_100_MET_NOT_CLINICAL_VALIDATION
```

All other statuses keep the product in clinician-reviewed mode.

## Paper Update After GO

If the run returns `GO_RETROSPECTIVE_BINARY_80_0_100_MET_NOT_CLINICAL_VALIDATION`, report
the retrospective held-out evidence without calling it clinically validated:

- report the untouched cohort source, size, positive-patient count, and hashes
- report coverage, specificity, missed positive patients, and exact 95% upper
  missed-risk bound
- keep the frozen thresholds and artifact version unchanged
- include the final decision JSON as the primary evidence artifact
- retain the clinical-validation NO-GO until a separate clinical study is complete

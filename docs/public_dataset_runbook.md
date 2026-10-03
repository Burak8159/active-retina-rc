# ActiveRetina-RC Public Dataset Runbook

This lane is for evidence that supports the upgraded ActiveRetina-RC flagship. It must not be reported as MFIDDR evidence or as four-field planning evidence unless the dataset truly has that structure.

## Evidence Roles

- Primary paired-view acquisition evidence: DeepDRiD.
- Additional two-view acquisition value: DRTiD if access is granted, and mBRSET if pairs are verified.
- Encoder/domain robustness: EyePACS, APTOS2019, BRSET, MMRDR, DDR/OIA, IDRiD, Messidor2.
- Specificity and disease-confounding checks: ODIR5K and RFMiD.
- Four-field CCG extension evidence: MFIDDR only unless another true four-field same-eye dataset is verified.

## Audit Available Local Data

```bash
PYTHONPATH=src python -m active_retina.audit_public_datasets \
  --registry configs/public_datasets.json \
  --json-output results/public_dataset_access_matrix.json \
  --md-output docs/public_dataset_evidence_matrix.md
```

## Generic Local Manifest

After a dataset is legally downloaded, create a canonical manifest from its metadata CSV:

```bash
PYTHONPATH=src python -m active_retina.prepare_generic_fundus \
  --dataset-name APTOS2019 \
  --csv data/APTOS2019/train.csv \
  --image-root data/APTOS2019/train_images \
  --image-column id_code \
  --grade-column diagnosis \
  --default-field fundus \
  --default-split train_labeled \
  --manifest-output results/APTOS2019_manifest.csv \
  --status-output results/APTOS2019_A0_public_status.json
```

If image filenames in the CSV omit extensions, create a small dataset-specific adapter rather than hand-editing labels.

## Reporting Rule

Every table must include a `dataset_role` column:

- `primary_four_field`: MFIDDR.
- `two_view_support`: DeepDRiD, DRTiD, mBRSET when paired fields are verified.
- `external_validation`: single-image or multimodal datasets.
- `specificity_check`: ODIR5K/RFMiD style multi-disease data.

The final paper decision remains:

- `FLAGSHIP_GO`: calibrated adaptive imaging saves acquisition burden while satisfying target missed-referable-DR risk on sealed patients, with external robustness/failure analysis strong enough for the final paper.
- `METHOD_ONLY_WEAK_EVIDENCE`: the method is valid, but savings, calibration power, or robustness are too weak for a flagship empirical claim.
- `NO_GO`: paired-view value or risk-controlled stopping fails.

Legacy MFIDDR-only planning labels:

- `GO-CCG`: four-field value, patient-specific individualization, and non-myopic planning value are all supported.
- `GO-MYOPIC`: four-field value and individualization exist, but non-myopic planning value is weak or untestable.
- `NO-GO`: multi-view value or individualization value is not empirically supported.

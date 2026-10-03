# PhysioNet mBRSET Access Action Pack

Status: `USER_CREDENTIAL_ACTION_REQUIRED`

PhysioNet states that mBRSET is a restricted-access resource. To access the
files, the user must:

1. Be a PhysioNet credentialed user.
2. Complete `CITI Data or Specimens Only Research`.
3. Sign the project Data Use Agreement.

I cannot complete training or sign the legal DUA on the user's behalf. Once the
account gate is complete, the rest of the ActiveRetina pipeline is automated.

## Why This Dataset Is Enough If Access Is Granted

The public mBRSET metadata was downloaded from the dataset authors' GitHub
repository and audited locally.

Result:

- 5,164 metadata rows
- 1,291 patients
- 4,859 quality-passing labeled rows
- 2,343 candidate complete same-eye suffix pairs
- 1,262 candidate paired patients
- 293 candidate referable-DR positive paired patients

This clears the required 59 positive-patient sample-size gate before image and
anatomy audit. The remaining issue is not cohort size. It is access to images
and suffix-to-anatomy confirmation.

Audit output:

`results/mbrset_public_metadata_candidate_summary.json`

## After Download

Place files like this:

```text
external/mBRSET/
  dataframe_brsetmobile.csv
  images/
    1.1.jpg
    1.2.jpg
    ...
```

Audit the suffix anatomy visually. Candidate maps:

```text
1:macula,2:optic_disc,3:macula,4:optic_disc
```

or

```text
1:optic_disc,2:macula,3:optic_disc,4:macula
```

Then run exactly one certification package command with the audited map.

Example:

```bash
PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.run_untouched_certification_package \
  --cohort-kind mbrset \
  --labels external/mBRSET/dataframe_brsetmobile.csv \
  --image-root external/mBRSET/images \
  --suffix-view-map 1:macula,2:optic_disc,3:macula,4:optic_disc \
  --attestation configs/untouched_attestation_template.json \
  --output-dir results/mbrset_untouched_certification
```

Only this final status upgrades the project:

```text
GO_FULL_AUTONOMOUS_COMPUTATIONAL_CERTIFICATE
```


# mBRSET Public Metadata Audit

Status: `PUBLIC_METADATA_FOUND_IMAGES_REQUIRED`

The mBRSET GitHub repository exposes a public metadata CSV named
`dataframe_brsetmobile.csv`. It includes:

- `patient`
- `file`
- `laterality`
- `final_quality`
- `final_icdr`
- demographic and clinical covariates

The public CSV does not include an explicit anatomical-view column. The file
pattern is structured as four images per patient in many rows:

- `<patient>.1.jpg`, right eye
- `<patient>.2.jpg`, right eye
- `<patient>.3.jpg`, left eye
- `<patient>.4.jpg`, left eye

This is almost enough for paired-eye certification, but not enough by itself.
The missing item is the suffix-to-anatomy mapping:

```text
1/3 = macula and 2/4 = optic_disc
```

or

```text
1/3 = optic_disc and 2/4 = macula
```

The importer now supports an explicit audited map:

```bash
--suffix-view-map 1:macula,2:optic_disc,3:macula,4:optic_disc
```

Use this only after manual inspection of downloaded images or official dataset
confirmation. Without that audit, mBRSET remains useful for external robustness
but not for a final paired-view autonomous certificate.

Once the images are available, manually inspect a random sample across both eyes
and multiple grades, record the audit, then run the one-shot certification
package with the audited map.


# External Untouched Data Routes

Status: `DATA_ACCESS_REQUIRED_FOR_FULL_AUTONOMOUS_CERTIFICATION`

The computation stack is frozen and ready. The missing item for a full
autonomous diagnosis claim is a genuinely untouched paired-view cohort.

## Best Route: mBRSET

mBRSET is the best currently identified public-data route for a real-world,
portable-camera certification cohort.

Why it matters:

- 5,164 fundus images.
- 1,291 patients.
- 997 referable-DR images reported in the dataset descriptor.
- ICDR labels.
- Captured with a portable handheld retinal camera.
- Images may be macula-centered or optic-disc-centered.
- Credentialed PhysioNet access with DUA is required.

Boundary:

The public descriptor lists image-level labels and says the image viewpoint can
be macula-centered or optic-disc-centered. A final paired-view certification run
requires local metadata to identify same-eye macula/disc pairs. If the downloaded
metadata does not expose viewpoint or pair structure, mBRSET can support
portable-camera external robustness but not paired-view autonomous certification.

Required after access:

1. Download mBRSET legally under the DUA.
2. Inspect metadata for patient, laterality, image file, ICDR grade, quality,
   and view position.
3. Build a canonical untouched certification manifest only from complete
   same-eye macula/disc pairs.
4. Run `validate_untouched_cohort.py`.
5. If eligible, run `certify_autonomous_candidate.py` once with no threshold
   repair.

## Other Routes

### DRTiD

Potentially excellent because it is a two-field macula/disc dataset. The current
known barrier is access/application status. If obtained, it can be converted
directly into the canonical paired-view certification manifest.

### Newly Collected Hardware Cohort

Best product-aligned route if the goal is deployment with your own acquisition
hardware. Use the raw paired-eye CSV template and require independent labels
before running certification. Do not inspect labels before freezing or repairing
the model.

### Single-Image DR Datasets

EyePACS, APTOS, DDR, IDRiD, Messidor, BRSET, and RFMiD can support robustness,
fairness, and single-image diagnostic validation. They cannot certify the
paired-view acquisition product unless same-eye macula/disc paired views are
available and verifiable.


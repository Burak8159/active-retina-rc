# ActiveRetina A0-A6 Protocol

This protocol freezes the feasibility phase before any CCG implementation. It follows the 2026-10-03 updated ActiveRetina source-of-truth spec.

## A0 Dataset Access and Schema

MFIDDR must be verified from local files before any primary empirical claim is made. The public repository states that MFIDDR contains 4,344 patients and 34,452 JPEG fundus images with DR grade labels and age, split into `train`, `test`, `train_fourpic_label.csv`, and `test_fourpic_label.csv`.

Field mapping:

- `F1`: macula-centered.
- `F2`: optic-disc-centered.
- `F3`: above the horizontal line on the optic disc.
- `F4`: below the horizontal line on the optic disc.

Filename mapping used by the pipeline:

- Right eye: `01->F1`, `02->F2`, `03->F3`, `04->F4`.
- Left eye: `05->F1`, `06->F2`, `07->F3`, `08->F4`.

Until the full files are locally present, A0 is blocked and all downstream primary metrics are unreported.

## A1 Immutable Splits

Create immutable patient-level partitions:

- train
- calibration
- validation
- sealed test

Never split images from the same patient across train/test. Keep both eyes from a patient in one partition unless a documented exception is approved. Preserve official patient-level splits when valid.

## A2 Baseline Encoder and Subset Classifier

Use a strong but simple pretrained encoder and a permutation-aware subset classifier. Full paper experiments should use exactly two backbone families:

- a research-strength modern pretrained/self-supervised encoder
- a lightweight deployment encoder such as ConvNeXt-Tiny, EfficientNetV2-S, or equivalent

The backbone is infrastructure; the paper is not a custom-ViT paper.

## A3 Field Subset Evaluation

Evaluate all relevant field subsets:

- 1 field
- 2 fields
- 3 fields
- all 4 fields
- all 15 non-empty subsets where practical

Report fixed macula-first subsets separately because Paper V1 fixes the first field as macula-centered. The policy may then choose optic disc, superior, inferior, or STOP.

## A4 Multi-View Value

Compare full multi-field performance against one-field performance using the primary endpoint: referable DR, grade >= 2.

Primary metrics:

- AUROC
- AUPRC
- sensitivity
- specificity
- Brier score / calibration
- mean number of retinal fields acquired

If additional fields do not materially improve performance, issue `NO-GO`.

## A5 Individualization Gap

At each acquisition budget `k`, compare:

- best universal fixed subset
- hindsight patient-specific best subset

Define:

`G_k = M_oracle(k) - M_fixed(k)`

This oracle is an upper bound, not a deployable baseline. If patient-specific oracle subsets do not materially outperform fixed subsets at equal budget, issue `NO-GO`.

## A6 Greedy-vs-Non-Myopic Gap

Compare:

- greedy one-step hindsight acquisition
- future-aware hindsight acquisition

If non-myopic planning materially beats greedy selection, issue `GO-CCG`.

If personalization exists but non-myopic planning does not materially beat greedy selection, issue `GO-MYOPIC`.

If personalization itself fails, issue `NO-GO`.

## Decision Labels

- `GO-CCG`: A4 and A5 pass, and A6 shows material non-myopic planning value.
- `GO-MYOPIC`: A4 and A5 pass, but A6 does not show material non-myopic value.
- `NO-GO`: A0 is blocked, or the empirical premise fails at A4/A5.

Do not implement CCG until a real-data A0-A6 decision is `GO-CCG`.

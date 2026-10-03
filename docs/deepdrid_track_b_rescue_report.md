# DeepDRiD Track-B Rescue Report

Status: **GO-DEEPDRID-TRACK-B-VALUE-EVIDENCE / HOLD-FULL-ACTIVERETINA-CCG**.

The backbone is frozen as infrastructure: frozen ResNet-50 ImageNet embeddings plus logistic heads from the existing full-CV pipeline. The paper focus returns to patient-specific sequential acquisition.

## suffix1_initial

- First-view AUROC: 0.848; two-view AUROC: 0.905.
- First-view Brier: 0.180; two-view Brier: 0.141.
- Mean second-view Brier reduction: 0.0390; helped: 0.596; harmed: 0.402; near-zero: 0.600.
- Best always_stop: views=1.000, AUROC=0.848, Brier=0.182.
- Best always_acquire: views=2.000, AUROC=0.897, Brier=0.148.
- Best entropy_acquire: views=1.509, AUROC=0.881, Brier=0.153, acquire_rate=0.509.
- Best uncertainty_acquire: views=1.404, AUROC=0.879, Brier=0.153, acquire_rate=0.404.
- Best trained_value_policy: views=1.876, AUROC=0.904, Brier=0.138, acquire_rate=0.876.

## suffix2_initial

- First-view AUROC: 0.887; two-view AUROC: 0.905.
- First-view Brier: 0.157; two-view Brier: 0.141.
- Mean second-view Brier reduction: 0.0159; helped: 0.535; harmed: 0.462; near-zero: 0.628.
- Best always_stop: views=1.000, AUROC=0.884, Brier=0.158.
- Best always_acquire: views=2.000, AUROC=0.897, Brier=0.148.
- Best entropy_acquire: views=1.511, AUROC=0.903, Brier=0.145, acquire_rate=0.511.
- Best uncertainty_acquire: views=1.396, AUROC=0.901, Brier=0.148, acquire_rate=0.396.
- Best trained_value_policy: views=1.949, AUROC=0.904, Brier=0.138, acquire_rate=0.949.

Interpretation: DeepDRiD supports adaptive acquisition-depth/value-of-second-view evidence. It does not support four-field next-view selection or non-myopic CCG; those remain MFIDDR-controlled. Mapping sensitivity remains important because first-view results differ between suffix branches, while two-view results are stable.

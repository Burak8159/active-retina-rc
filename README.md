# ActiveRetina Flagship Paper

Status: `GO_CLINICIAN_REVIEWED_COMPUTATIONAL_PRODUCT`

Autonomous status:
`CONDITIONAL_GO_AUTONOMOUS_DIAGNOSIS_AFTER_UNTOUCHED_CERTIFICATION`.

Software product status:
`GO_CLINICIAN_REVIEWED_COMPUTATIONAL_PRODUCT`.
The repository now includes a locked model artifact, fail-closed clinical
gateway, real image case runner, local browser station, audit hashes, JSON
API, HTML clinician report, and hardware-enclosure runbook. This is
product-grade workflow software for supervised clinical review, not autonomous
diagnosis.

The current evidence package supports a transparent arXiv benchmark/methods
preprint. It is not a clinical deployment claim. IEEE journal submission remains
conditional on a stronger DeepDRiD-only finite-sample risk-control path. See
`docs/flagship_evidence_layer.md`.

The frozen autonomous candidate is computation-ready. The final autonomous
claim is controlled by one command:

`PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.run_untouched_certification_package --cohort-kind mbrset --labels external/mBRSET/<metadata-file>.csv --image-root external/mBRSET/<image-folder> --attestation configs/untouched_attestation_template.json --output-dir results/mbrset_untouched_certification`

Only `GO_FULL_AUTONOMOUS_COMPUTATIONAL_CERTIFICATE` upgrades the project to
autonomous-certified. See `docs/full_autonomous_certification_runbook.md`.

Prior policy results are exploratory and uncertified; the original test holdout
has been exposed. See `docs/evidence_integrity_audit.md`. Corrected development
runs use training patients only and exclude test evaluation by default.

This folder contains the reproducible scaffold for the upgraded ActiveRetina flagship paper:

**Risk-Controlled Adaptive Retinal Imaging for Diabetic Retinopathy Screening**

Current evidence summary: the dependency-light executable baseline and the frozen ImageNet-pretrained ResNet-50 feature baseline have both run end to end on DeepDRiD. The flagship evidence layer now gives paired-view discrimination evidence, risk-coverage-cost baselines, patient-cluster bootstrap uncertainty, mapping sensitivity, mechanism diagnostics, generated figures, a machine-readable GO decision for a benchmark preprint, and an exact finite-sample calibration risk audit. The study remains limited by unresolved DeepDRiD suffix-to-field mapping, exposed historical test data, small validation size, and missing five-class prediction outputs. See `docs/flagship_evidence_layer.md`.

The project no longer waits for request-only datasets. The flagship contribution is now safety-first paired-view adaptive imaging:

- decide when one retinal image is enough,
- decide when another retinal view is medically useful,
- defer when the model cannot satisfy a target missed-referable-DR risk level,
- report acquisition savings only at calibrated, patient-level operating points.

## Clinical Workflow Software

Train the locked software artifact:

`PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.train_clinical_artifact`

Run the fail-closed clinical gateway:

`PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.clinical_gateway --input configs/clinical_gateway_example.json --output results/clinical_gateway_example_output.json`

Run a real paired-image case through the product workflow:

`PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.clinical_case_runner --case-id DEMO-IMG-001 --first-image external/DeepDRiD_v1_1/deepdrdoc-DeepDRiD-56d8af7/regular_fundus_images/regular-fundus-training/Images/1/1_l2.jpg --second-image external/DeepDRiD_v1_1/deepdrdoc-DeepDRiD-56d8af7/regular_fundus_images/regular-fundus-training/Images/1/1_l1.jpg --output-json results/clinical_case_demo_output.json --output-html results/clinical_case_demo_report.html`

The demo case output uses the frozen feature cache, writes SHA-256 input hashes,
and produces `results/clinical_case_demo_report.html`. Every case is marked
`manual_review_required: true`.

Run the live encoder path for new imported images:

`HF_HOME=.model_cache/huggingface TORCH_HOME=.model_cache/torch PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.clinical_case_runner --case-id DEMO-LIVE-ENCODER-001 --first-image external/DeepDRiD_v1_1/deepdrdoc-DeepDRiD-56d8af7/regular_fundus_images/regular-fundus-training/Images/1/1_l2.jpg --second-image external/DeepDRiD_v1_1/deepdrdoc-DeepDRiD-56d8af7/regular_fundus_images/regular-fundus-training/Images/1/1_l1.jpg --force-live-encoder --output-json results/clinical_case_live_encoder_output.json --output-html results/clinical_case_live_encoder_report.html`

Run the local browser station:

`HF_HOME=.model_cache/huggingface TORCH_HOME=.model_cache/torch PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.clinical_station --host 127.0.0.1 --port 8765`

Then open `http://127.0.0.1:8765`.

Analysis API:

`POST http://127.0.0.1:8765/api/analyze`

API contract:

`docs/api_contract.md`

Hardware/enclosure workflow:

`docs/hardware_enclosure_clinical_station.md`

## Flagship Thesis

Existing diabetic retinopathy models usually predict from whatever images are already available. ActiveRetina-RC instead treats imaging as a clinical decision process: after an initial retinal photograph, the system must choose `STOP`, `ACQUIRE`, or `DEFER` under an explicit missed-disease risk constraint.

The intended contribution is risk-controlled acquisition under uncertainty and
dataset shift. The final paper is a DeepDRiD-only decision-theoretic acquisition
benchmark, not a private-dataset access project.

## Primary Decisions

- Primary available acquisition dataset: DeepDRiD paired color fundus views.
- Primary endpoint: referable DR, defined as grade >= 2 where a 5-class grade is available.
- Secondary endpoint: five-class DR grading, grades 0-4.
- Safety target: missed referable DR risk, calibrated at patient level.
- Core policy actions: `STOP`, `ACQUIRE_SECOND_VIEW`, `DEFER`.
- Request-only datasets: out of scope for this paper.
- CCG status: out of scope. Non-myopic four-field planning is no longer a dependency.

## Flagship Gates

- F0: Dataset/legal/schema audit for DeepDRiD plus every external dataset used.
- F1: Immutable patient-level splits with sealed test patients.
- F2: Cross-fitted single-view and two-view predictions.
- F3: Paired single-view versus two-view value with bootstrap confidence intervals.
- F4: Calibrated stop/acquire/defer policies satisfying target missed-referable-DR risk on calibration patients.
- F5: Sealed-test cost-risk curves: sensitivity, specificity, AUPRC, AUROC, Brier score, defer rate, and mean images acquired.
- F6: External robustness and shift analysis on public DR datasets, clearly separating acquisition evidence from single-image robustness evidence.
- F7: Final decision: `GO_DEEPDRID_ARXIV_BENCHMARK_CONDITIONAL_IEEE`, `METHOD_ONLY_WEAK_EVIDENCE`, or `NO_GO`.

## Run Order

1. Verify DeepDRiD manifests, local image files, and suffix-mapping sensitivity branches. Done.
2. Generate cross-fitted predictions for both mapping branches with the dependency-light executable baseline. Done.
3. Evaluate one-view, two-view, and hindsight second-view value using only out-of-fold/held-out predictions. Done.
4. Implement calibrated stop/acquire/defer policies. Done.
5. Evaluate sealed-test cost-risk curves. Done for the executable baseline and pretrained ResNet-50 feature baseline.
6. Resolve DeepDRiD suffix mapping or make the paper suffix-agnostic before final field-named claims.
7. Add formal finite-sample risk-control analysis from calibration patients.
8. Generate V2 failure-analysis artifacts:

`PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.rebuild_v2_artifacts`

This creates grade-stratified referable analysis and paired-score delta tables
from existing predictions. It does not complete the five-class secondary
endpoint because the current prediction exports do not contain five-class class
probabilities or predicted grades.

9. Write the IEEEtran manuscript from frozen real metrics only after the relevant
readiness gate is satisfied.

The encoder/classifier scaffold lives in `src/active_retina/modeling.py`. It uses a pretrained ConvNeXt-Tiny candidate with ResNet-50 fallback and a permutation-aware set-pooling head.

## Public Supporting Datasets

Run the public evidence audit:

`PYTHONPATH=src python -m active_retina.audit_public_datasets --registry configs/public_datasets.json`

This creates `results/public_dataset_access_matrix.json`, `results/public_dataset_access_matrix.csv`, and `docs/public_dataset_evidence_matrix.md`. Public datasets are assigned evidence roles and cannot be promoted beyond what their schema supports.

For locally downloaded datasets with a metadata CSV, use:

`PYTHONPATH=src python -m active_retina.prepare_generic_fundus ...`

See `docs/public_dataset_runbook.md` for examples and reporting rules.

## Non-Negotiables

- Test patients remain sealed.
- Patient IDs, not images, are the split unit.
- Any learned counterfactual target must be cross-fitted.
- Fixture smoke outputs must keep `TEST_FIXTURE` in filenames and reports.
- No paper claim may be made from fixture outputs.
- Mapping sensitivity is mandatory until DeepDRiD suffix-to-field identity is officially verified.
- Dependency-light baseline outputs prove pipeline readiness, not final model superiority.
- Pretrained ResNet-50 outputs are real experimental evidence, but current policy claims remain blocked by mapping sensitivity and external robustness gaps.

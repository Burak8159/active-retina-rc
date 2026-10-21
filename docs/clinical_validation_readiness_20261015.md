# ActiveRetina Clinical-Validation Readiness

Assessment date: 2026-10-15

## Decision

`NOT_CLINICALLY_VALIDATED`; `NO_GO_FOR_AUTONOMOUS_CLINICAL_DIAGNOSIS`.

This is an evidence and study-readiness record, not a claim of clinical
validation. No model, retrospective score, or manuscript edit can replace an
independent clinical study.

## Current model/evidence objects

These are different systems and must not be merged into one headline metric:

| Object | What it is | Current evidence | Boundary |
|---|---|---|---|
| 70.13 score-policy replay | Deterministic replay of a policy over precomputed multi-source scores | 270/385 negatives cleared (70.13%); exposed records informed policy selection | Not image-model performance, not an independent test, not a risk guarantee |
| Paired-view candidate | ResNet-50 `resnet50.a1_in1k` frozen image features plus a trained balanced logistic head; low/high thresholds emit autonomous-candidate labels and defer the middle | 396 training patients / 792 eyes in artifact metadata; thresholds selected using inspected development data | Selective candidate, not full binary coverage; DeepDRiD view anatomy remains a mapping concern |
| MMRDR-UWF image model | ImageNet-pretrained ResNet-50 frozen encoder plus a trained six-grade linear head; threshold selected on its validation split | Frozen official test: 1,288 positive image rows, 5 missed; specificity 262/1,309 = 20.02%; binary action coverage 100%; one-sided 95% image-row miss-risk UCB 0.814% | Held-out test within one dataset, not prospective clinical validation; poor specificity and nonzero misses; exported table lacks patient IDs for patient-level reanalysis |
| Clinical station artifact | Local workflow software using cached feature/model artifacts | Workflow tests and user interface exist in the repository | Software functioning does not establish diagnostic validity or clinical benefit |

Artifact hashes captured in this audit (SHA-256):

- Paired-view candidate: `97f720e1ed1516a691f5a6fded66b891856349f9e64cd7c84ea84bd67668fb31`
- Paired-view metadata: `03f9a5448859403cb0f40122a61f840f4b9ace79d44b29088e9211dbf8e9dd73`
- MMRDR-UWF checkpoint: `2357888995465d7b8dff03b8ac49b294fd94e17868a5aafdb403d5a26084c32b`
- MMRDR-UWF official-test summary: `b8a9f1fa0ba5f2274104bc4e2a096c93e5fc060771a548d21748c13deb4e10f5`

The MMRDR-UWF test is now exposed because it has been evaluated. Do not tune on
it and then reuse it as confirmatory evidence. The five false-negative image
rows rule out any zero-observed-miss claim for that run.

## Intended-use statement: draft, not yet locked

Proposed study question: among people undergoing diabetic-retinopathy
screening, can a frozen fundus-image system assign a screening disposition for
referable DR at a prespecified specificity while maintaining a prespecified
miss-risk bound?

Before confirmatory data access, a qualified clinical lead must resolve and
sign off on the population, care setting, camera/input types, one-image versus
paired-image workflow, definition of referable disease, reference grading and
adjudication, what constitutes an ungradable image, and whether the output is
screening triage or a diagnosis. Until signed, this statement is not a final
intended-use claim. The current paired-view model and MMRDR-UWF model have
different inputs and cannot share one intended-use claim.

## Frozen computational research gate

The certification runner has been corrected so a conditional-risk result
cannot be presented as full binary autonomy when the policy defers cases. Its
`80/0/100` research gate requires all of the following on the same untouched
cohort:

- 100% of eligible eye rows receive a binary action; a defer/abstention fails
  full binary coverage.
- Zero false-negative referable eye rows and zero missed referable patients.
- At least 59 fully decided referable-positive patients and a one-sided exact
  95% upper bound on patient-level missed risk no greater than 5%.
- Patient-level specificity of at least 80%; report its confidence interval
  and the full confusion counts, not only the point estimate.
- Model, preprocessing, thresholds, inference code, cohort manifest, and
  analysis plan are hash-recorded before predictions are joined to labels.

Passing this gate would mean only that the locked computational target was
met on that cohort. It would not establish prospective benefit, clinical
workflow safety, transportability to other devices/sites, or regulatory
authorization.

## Study needed for clinical validation

1. **Select one viable candidate using development data only.** The MMRDR-UWF
   result is not a candidate for an 80%-specificity/zero-miss claim. The paired
   candidate currently defers and is not full-binary. Do not pick between
   candidates after viewing confirmatory results.
2. **Lock the clinical protocol.** A clinician/data custodian specifies the
   intended population, device/input contract, thresholds, reference standard,
   eligibility, handling of both eyes, ungradable images, missing images, and
   failed inference. Pre-register endpoints and the statistical analysis plan.
3. **Obtain a new representative cohort.** Enroll or access consecutive or
   otherwise clearly sampled patients from a site/domain not used for model
   development. Keep all patient records together. Preserve enough positive
   and negative patients for prespecified sensitivity, specificity, subgroup,
   and calibration precision. The 59-positive/zero-miss calculation is only a
   minimal one-sided binomial risk-bound condition, not a complete study-size
   calculation.
4. **Establish the reference standard independently.** Use qualified graders
   masked to model output, a prespecified grading protocol, and adjudication
   for disagreements. Record gradability and label missingness.
5. **Run once with the artifact frozen.** A custodian holds labels or the
   cohort until the model and analysis hashes are deposited. Execute inference
   without threshold retuning or manual exclusion of difficult cases. Record
   every decode, preprocessing, and inference failure.
6. **Report all primary and safety outcomes.** Include patient- and eye-level
   confusion counts; sensitivity/missed-disease rate and exact confidence
   bounds; specificity and confidence interval; coverage; ungradable and
   failure rates; calibration if probabilities are used; subgroup results;
   and a blinded error review. If the intended claim is clinical utility,
   evaluate workflow and patient-care impact, not only offline accuracy.
7. **Complete the study-appropriate reporting.** Apply [CLAIM 2024](https://pubs.rsna.org/doi/10.1148/ryai.240300) to
   medical-imaging AI reporting; [STARD-AI](https://www.nature.com/articles/s41591-025-03953-8) to diagnostic-accuracy reporting;
   [DECIDE-AI](https://www.bmj.com/content/377/bmj-2022-070904) to early live clinical evaluation; and
   CONSORT-AI/SPIRIT-AI if the work is an AI clinical trial. These are
   reporting guidance, not substitutes for study conduct or validation.

## Current blockers and owners

| Blocker | Evidence needed to close it | Owner |
|---|---|---|
| No eligible untouched cohort | New cohort, dataset provenance, exact-overlap audit, custodian attestation | Clinical/data partner |
| No viable full-binary candidate demonstrated | Development-only model selection meeting the locked performance objective, with no use of confirmatory labels | Model team |
| Intended use and reference standard not finalized | Signed study protocol from qualified clinical lead | Clinical lead / study owner |
| Current external-like test evidence is unfavorable | New candidate and a separate untouched evaluation; MMRDR test stays exposed | Model team + custodian |
| Patient-level MMRDR analysis unavailable from released predictions | Permitted patient-linked evaluation table or new patient-level cohort | Dataset owner / custodian |
| Live workflow and benefit untested | Prospective or otherwise appropriate clinical workflow evaluation | Clinical site / study team |

## Reproducibility and integrity controls now in code

- The untouched-cohort validator checks identifiers in recorded DDR,
  DeepDRiD, DRTiD, IDRiD, MMRDR and MFIDDR-fixture manifests using dataset
  namespaces, and checks exact image-content hashes where local files exist.
- A data-owner attestation remains mandatory; code cannot prove that labels
  were never inspected or that two differently named IDs are the same person.
- The certification gate distinguishes conditional risk certification from
  full-binary coverage and the 80% specificity target.
- A research certificate is never emitted as `clinical_validation_status`.

## Final status

The repository now has a stricter computational gate and a concrete clinical
validation protocol. **The model is not clinically validated, no candidate
currently meets the full-binary clinical target, and the required independent
clinical cohort/workflow study has not been run.** The remaining study cannot
be completed locally from the current retrospective files; it requires a
clinical/data partner and new data.

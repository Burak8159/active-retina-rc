# ActiveRetina Flagship Rebuild V2

Date: `2026-10-03`

Status: `REBUILD_SOURCE_OF_TRUTH`

## Decision

The current DeepDRiD-only manuscript is not the paper. It is a development artifact.

The flagship paper is rebuilt as:

**ActiveRetina: Risk-Controlled Sequential Retinal Evidence Acquisition for Diabetic Retinopathy Screening**

The paper should not try to beat MFIDDR by pretending to have a larger or cleaner four-field dataset. MFIDDR's strongest contribution is dataset scale and standardized same-eye multi-field coverage. The new paper must be stronger in a different dimension: it must turn retinal imaging from a fixed-input grading benchmark into a clinically constrained decision problem.

## What "Stronger Than MFIDDR" Means

MFIDDR-style contribution:

- introduces a large four-field same-eye dataset,
- benchmarks multi-view DR grading,
- shows multi-field information improves grading.

ActiveRetina V2 contribution:

- learns when retinal evidence is enough,
- explicitly trades acquisition burden, missed referable DR, deferral, and workflow cost,
- calibrates policies with patient-level risk constraints,
- separates acquisition evidence from external robustness evidence,
- tests failure modes across public datasets,
- reports when evidence is insufficient rather than forcing a positive result.

This is stronger as a paper only if the final manuscript contains evidence beyond a dataset benchmark. It must be a decision-theoretic and clinically framed study, not a smaller substitute for MFIDDR.

## Non-Negotiable Boundaries

- Do not claim four-field or non-myopic field planning without MFIDDR, DRTiD, or another true same-eye multi-field dataset.
- Do not call a manuscript arXiv-ready or IEEE-ready until the readiness gates below are actually satisfied.
- Do not use the exposed historical DeepDRiD test partition for confirmatory claims.
- Do not treat single-image datasets as acquisition-policy evidence.
- Do not treat DeepDRiD suffix mapping as official unless verified from source files or authors.
- Do not optimize language to sound stronger than the data.

## New Paper Thesis

Fixed retinal imaging protocols waste acquisition effort on easy eyes and may still be unsafe on uncertain eyes. ActiveRetina formulates DR screening as sequential evidence acquisition:

1. acquire an initial retinal image,
2. estimate whether the current evidence is sufficient under a missed-referable-DR risk target,
3. either stop, acquire an additional view, or defer to full imaging/human review,
4. report the cost-risk-defer frontier rather than a single accuracy number.

## Evidence Ladder

### Tier 0: Infrastructure Smoke Evidence

Purpose: prove the pipeline works, not the science.

Acceptable data:

- MFIDDR test fixture,
- schema-only metadata,
- tiny local subsets.

Allowed claims:

- scripts run,
- split manifests are patient-disjoint,
- tables/plots can be generated.

Forbidden claims:

- scientific performance,
- acquisition benefit,
- clinical safety.

### Tier 1: DeepDRiD Paired-View Development Evidence

Purpose: establish the first real acquisition-depth result.

Required outputs:

- two suffix-mapping branches,
- patient-level train/calibration/validation splits,
- first-view, second-view, and paired-view performance,
- conservative stop/acquire/defer policy,
- zero-deferral ablation,
- bootstrap intervals,
- grade-stratified and quality-stratified errors if metadata permits,
- explicit statement that this is not confirmatory.

Minimum finding needed:

- paired views improve discrimination or calibration in validation,
- calibrated policy has a useful acquisition/defer frontier,
- conclusions are stable enough under suffix swap to justify continuing.

### Tier 2: External Public Robustness Evidence

Purpose: prevent the paper from being a one-dataset curiosity.

Dataset roles:

- BRSET / mBRSET: patient-level and demographic robustness when accessible.
- MMRDR: modality and lesion-aware robustness.
- DDR / IDRiD: lesion-level failure analysis.
- APTOS / EyePACS / Messidor2: broad single-image DR discrimination stress tests.
- ODIR / RFMiD: specificity under other retinal disease labels.

Allowed claims:

- external discrimination,
- calibration shift,
- defer behavior under domain shift,
- lesion/quality failure modes.

Forbidden claims:

- field acquisition benefit from single-image data,
- non-myopic planning,
- replacement for paired validation.

### Tier 3: Confirmatory Paired or Multi-Field Evidence

Purpose: make the paper genuinely strong.

Preferred datasets:

- MFIDDR, if access arrives,
- DRTiD, if access is granted,
- mBRSET only if paired macula/disc images and labels are verified.

Required outputs:

- frozen policy before confirmatory labels are inspected,
- patient-disjoint held-out evaluation,
- prespecified risk and specificity endpoints,
- finite-sample uncertainty,
- full comparison against fixed-view, selective-classification, and acquisition policies.

This tier is the difference between a serious development preprint and a strong journal submission.

## Stronger-Than-MFIDDR Experiment Stack

### A. Dataset and Schema Audit

For every dataset:

- legal access status,
- license/terms,
- patient/eye identifiers,
- laterality,
- DR label definition,
- image-field semantics,
- image quality labels,
- lesion labels,
- demographic metadata,
- train/test split authority,
- whether it supports acquisition evidence or robustness only.

### B. Encoder Families

At minimum:

- ImageNet ResNet-50 frozen encoder,
- ImageNet ViT or ConvNeXt frozen encoder,
- ophthalmic foundation model if accessible locally and legally,
- optional fine-tuning only after frozen baselines are stable.

Report:

- identical splits,
- identical calibration protocol,
- compute budget,
- failure cases.

### C. Acquisition Policies

Baselines:

- always one view,
- always two views,
- first-view confidence triage,
- fixed second-view acquisition,
- random acquisition when multiple views exist,
- best fixed subset at each budget for multi-field data.

ActiveRetina policies:

- conservative stop/acquire/defer,
- zero-deferral ablation,
- cross-fitted second-view value oracle,
- learned acquisition policy only after oracle gap exists,
- non-myopic/CCG only if true multi-step data exist.

### D. Statistical Standard

Required:

- patient-cluster bootstrap,
- calibration/validation/test split discipline,
- no threshold tuning on test,
- risk target and specificity target fixed before evaluation,
- confidence intervals for risk, specificity, image count, defer rate, and cost.

For journal submission:

- formal finite-sample risk bounds or conformal/selective-risk framing,
- sensitivity to threshold choice,
- subgroup and grade-stratified error analysis,
- multiple-comparison discipline for many policies.

### E. Clinical Utility Model

The paper must go beyond "fewer images":

- image acquisition cost,
- deferral cost,
- false-positive referral cost,
- missed referable DR cost,
- technician/workflow burden,
- sensitivity analysis over cost ratios,
- policy frontiers instead of a single operating point.

### F. Failure Analysis

Required:

- grade-stratified errors,
- image-quality-stratified errors,
- lesion-presence stratification where available,
- domain-shift calibration plots,
- examples of cases requiring second view,
- examples of cases deferred by a risk-first policy.

## Readiness Gates

### Gate A: arXiv-Development Preprint

All must be true:

- DeepDRiD paired-view analysis is complete and reproducible.
- Tables and plots are clean.
- Suffix mapping is either verified or permanently framed as sensitivity analysis.
- Related work covers active acquisition, selective prediction, DR grading, and multi-view retinal learning.
- Claims are explicitly development-only.
- Secondary analyses are either complete or removed from the abstract/contribution list.

Decision label: `ARXIV_DEVELOPMENT_READY`.

### Gate B: Strong arXiv / Workshop / Conference Submission

All Gate A items, plus:

- at least one external public dataset robustness analysis,
- stronger encoder baseline,
- failure analysis,
- cost-risk-defer curves,
- comparison to confidence triage and acquisition baselines.

Decision label: `STRONG_PREPRINT_READY`.

### Gate C: IEEE Journal Candidate

All Gate B items, plus:

- clean unexposed paired/multi-field confirmatory evaluation,
- or an exceptionally strong multi-dataset public robustness package with transparent limitation that acquisition confirmation remains paired-data-limited,
- formal risk-bound or selective-risk analysis,
- full related-work coverage,
- complete secondary five-class grading analysis,
- mature clinical workflow framing.

Decision label: `IEEE_CANDIDATE`.

### Gate D: Flagship Journal Submission

All Gate C items, plus:

- confirmatory paired or multi-field validation,
- stable findings across mapping assumptions and encoder families,
- statistically credible uncertainty,
- clinically interpretable cost-risk tradeoffs,
- no exposed-test dependence,
- no unresolved primary endpoint ambiguity.

Decision label: `FLAGSHIP_SUBMISSION_READY`.

## Immediate Work Order

1. Freeze the current manuscript as `development_draft`.
2. Create a new evidence-first manuscript outline that does not claim readiness.
3. Complete the DeepDRiD secondary five-class analysis or explicitly remove it.
4. Generate grade-stratified and policy-frontier plots from existing predictions.
5. Audit which public datasets are immediately accessible without new approvals.
6. Add one external robustness dataset only if legal access and labels are confirmed.
7. Re-run the readiness gate after new evidence, not before.

## Current Decision

`METHOD_DEVELOPMENT_CONTINUE`

The paper can become stronger than an MFIDDR-only benchmark only by becoming a risk-controlled, multi-dataset, decision-theoretic study. It is not there yet.

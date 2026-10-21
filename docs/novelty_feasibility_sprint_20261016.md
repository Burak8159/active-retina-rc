# Novelty Feasibility Sprint (2026-10-16)

## Decision

Do not claim a new model, estimator, or general theorem. Prior work already establishes that model selection can overfit finite validation criteria, that nested validation is a standard mitigation, and that screening test sets should be independent and reused sparingly. The defensible candidate contribution is a **case-study audit of operating-point portability under two distinct perturbations: patient-group split realization and candidate-score-panel construction**.

This is a narrow empirical contribution, not yet a flagship claim. Its value is that it tests whether a tempting operational conclusion (a score policy that clears many negatives with zero observed misses) survives a more explicit development-selection procedure and plausible changes to the available score panel.

## Falsifiable question

On a retrospectively exposed, multi-source diabetic-retinopathy score cohort, does a broader, label-unfiltered aligned score panel improve patient-grouped nested operating-point performance over a full-cohort label-screened contributor shortlist, consistently across paired split seeds, under the same miss-count constraint?

The prediction to test in a future independent study is that neither one favorable split nor a larger candidate panel is sufficient evidence of a stable high-specificity, low-miss operating point; candidate-panel effects will be smaller and less stable than the same-cohort replay suggests.

## Analysis executed on the saved artifacts

Source: `results/candidate_panel_seed_sweep_20261015.json`, cross-checked against the five per-seed run JSONs. Cohort: 981 eyes, 928 patients, 385 gradable nonreferable eyes. Both panels used the same five-outer/four-inner patient-grouped selector and paired seeds. The cohort and score artifact library had previously been inspected; these are development diagnostics only.

| Candidate panel | Mean specificity | Range across five seeds | Mean referable misses | Zero-miss runs |
|---|---:|---:|---:|---:|
| Full-cohort label-screened contributor shortlist | 57.45% | 49.09–60.78% | 2.6 | 0/5 |
| All complete aligned score columns | 56.31% | 54.81–57.40% | 4.2 | 0/5 |

The paired specificity contrast (all-aligned minus shortlist) averaged -1.14 percentage points and ranged from -4.16 to +8.31 points: all-aligned was lower in four seeds and higher in one. Neither panel achieved zero misses in any run. The 70.13% same-cohort replay is 13.82 points above the all-aligned mean, but this arithmetic difference is **not an unbiased estimate of optimism**: the replay and nested analysis use different selection procedures, and both use historically exposed evidence.

Under the post hoc descriptive filter of fewer than five misses, the maximum of the ten panel-by-seed outcomes was 60.78% specificity with three misses. It is selected after viewing those outcomes and must not be reported as a prospective or independent performance estimate.

## Prior-art check and novelty boundary

The central principles are established, not novel: Cawley and Talbot analyze selection bias from overfitting model-selection criteria; medical-imaging cross-validation guidance explains nested evaluation; CLAIM 2024 emphasizes patient-level partitioning; screening-test-set recommendations explicitly warn against repeated testing of multiple versions and selecting the best afterward. This project therefore should not claim to invent nested cross-validation, selection-bias correction, or independent-test-set practice.

Potential contribution wording, pending stronger replication:

> We present a provenance-aware empirical audit showing that, in one historically exposed multi-source DR development pipeline, same-cohort zero-miss replay did not persist under patient-grouped nested selection, and candidate-panel effects varied in direction across paired split seeds.

The phrase “in one ... pipeline” is essential. Current results do not establish prevalence of this phenomenon across medical AI, causal effects of panel size, future-patient performance, or clinical utility.

## What would materially strengthen it

1. Freeze the protocol and analysis code before touching a new cohort or score library; record hashes and a prespecified primary endpoint.
2. Repeat the paired panel/split experiment on at least one genuinely untouched cohort/library with patient identifiers and documented provenance. Do not describe an inspected source or rerun as untouched.
3. Add a synthetic controlled experiment with known ground-truth data-generating mechanisms to separate selection optimism from domain shift and prevalence/composition effects. Report it as simulation, never as clinical evidence.
4. Use patient-level uncertainty procedures that respect paired predictions and clustered eyes; report counts and intervals, and avoid treating five overlapping split seeds as five independent studies.
5. Compare against prespecified simple baselines and report the whole miss-specificity frontier, not the best post hoc point. The candidate-panel comparison alone is not a new method.

## Go/no-go for a flagship claim

**Current:** plausible retrospective audit contribution; novelty is narrow and not yet externally replicated. Appropriate as a transparent preprint/audit case study if all submission and data-use gates are cleared.

**Not supported:** “groundbreaking,” “clinically validated,” “autonomous diagnosis,” or a general claim that threshold selection causes a 13.82-point performance drop.

**Advance to a stronger journal contribution only if:** the prespecified effect or instability pattern replicates on untouched data, the controlled simulation verifies the proposed mechanism, and all evidence objects remain auditable. A null replication is a valid result and should narrow or stop the claim.

## Selected primary sources

- Cawley and Talbot, “On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation,” *JMLR* 11 (2010), 2079–2107. https://jmlr.csail.mit.edu/beta/papers/v11/cawley10a.html
- Mongan et al., “Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update,” *Radiology: Artificial Intelligence* (2024). https://doi.org/10.1148/ryai.240300
- Sounderajah et al., “Recommendations for the development and use of imaging test sets to investigate the test performance of artificial intelligence in health screening,” *The Lancet Digital Health* 4 (2022). https://doi.org/10.1016/S2589-7500(22)00186-8
- Roelofs et al., “A Guide to Cross-Validation for Artificial Intelligence in Medical Imaging,” *Radiology: Artificial Intelligence* (2023). https://doi.org/10.1148/ryai.220232

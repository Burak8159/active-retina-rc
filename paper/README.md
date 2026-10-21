# ActiveRetina Paper Package

Status checked 2026-10-19 (arXiv source v30 refresh). The arXiv manuscript is a retrospective evidence audit, not a clinically validated diagnostic model. The latest edit clarifies that paired miss outcomes are eye-level rates and patient clusters are only the resampling unit. It changes no result and adds no independent evidence.

## IEEE Review Snapshot

- Manuscript source: [`active_retina_ieee_so_far.tex`](active_retina_ieee_so_far.tex)
- Release gates and evidence boundaries: [`SUBMISSION_RELEASE_CHECKLIST.md`](SUBMISSION_RELEASE_CHECKLIST.md)
- Current aggregate-only IEEE review snapshot: [`submission/active_retina_ieee_review_bundle_2026-10-19_v115.zip`](submission/active_retina_ieee_review_bundle_2026-10-19_v115.zip)
- Verification receipt: [`submission/verification_2026-10-19_v115.json`](submission/verification_2026-10-19_v115.json)

## Main Evidence

The frozen MMRDR-UWF classifier detected 1,283/1,288 referable test images, missed five, and cleared 262/1,309 nonreferable images (20.02% image-level specificity; Wilson 95% CI 17.94--22.27%). The exact local manifest, prediction, and source-annotation joins agree on all 2,597 image rows and grades. The source annotations do not expose patient identifiers, so patient-clustered uncertainty and independent verification of patient-level separation are unavailable.

The post-hoc exact-byte audit scanned all 10,404 local UWF images, found eight duplicate groups and two groups crossing official train/test partitions. Both cross-split test images were positive, referred, and had exact copies in model-fit data. Excluding them descriptively leaves five misses and 20.02% specificity unchanged. This is a split-integrity sensitivity check, not a clean rerun or independent test; exact-byte hashing cannot detect near-duplicates or establish patient identity. The aggregate receipt and audit code are included in v101, but the original images and row-level records are not.

The 70.13% value is deterministic replay of a different score policy on an exposed 981-eye cohort. It is not raw-image model performance. Separately, a hash-checked post-hoc analysis of frozen MMRDR test scores found ROC-AUC 0.9173 and average precision 0.9234, but only 9.40% specificity at 100% observed sensitivity. These test-derived diagnostics are not deployable thresholds and do not change the model or its failed locked operating point.

A paired replay-versus-patient-grouped analysis on that same exposed cohort reports 70.13% (270/385) versus 69.61% (268/385) eye-level specificity, with a conditional paired interval of -1.31 to 0.00 points. Among 596 referable or ungradable eye records, replay had 0 missed required eyes and grouped calibration had 3; the paired eye-level missed-required-eye-rate difference is +0.50 points (conditional interval 0.00--1.17, including zero). Patient clusters are the resampling unit, not the endpoint. This is an audit of selection procedure sensitivity, not independent validation; the per-eye inputs are excluded from the public snapshots.

The v98 research analysis quantifies source-associated variability in the exposed replay: source-stratified patient-cluster intervals and exploratory pairwise contrasts are reported in both manuscripts. Equal-source reweighting raises the arithmetic summary by 6.55 points; the decomposition shows that this is driven by reweighting the small DeepDRiD subset, not by model improvement. The analysis cannot identify camera, case-mix, label, or algorithm causes and does not adjust for cohort exposure or policy selection.

## Reproduction Boundary

The arXiv source package is [`submission/active_retina_arxiv_source_2026-10-19_v30.zip`](submission/active_retina_arxiv_source_2026-10-19_v30.zip). It includes the current manuscript, selected audit code, and aggregate-only receipts, but omits local row-level score/label inputs and patient manifest, so the paired audit cannot be rerun from the archive. It does not reproduce MMRDR inference or the score-policy replay.

The v100 snapshot added an exact-byte split-overlap audit to the fixed-threshold MMRDR result. The v101 snapshot adds an explicit input/unit field to both selection-history tables, distinguishing image-level inference from eye-level score replay and case-level simulation. The v102 snapshot adds paired replay-versus-grouped transition counts and conditional cluster uncertainty. These are reporting/audit improvements, not model changes or independent validation. Row-level score data, predictions, images, and checkpoints remain excluded.

The v99 snapshot repaired the pooled bootstrap implementation and aligned the abstract operating-point comparison. The v98 snapshot adds conditional source-specific patient-cluster intervals, exploratory pairwise source contrasts, and a transparent arithmetic decomposition of source-mixture reweighting. These do not change model outputs or establish external validity.

The previous v90 snapshot added verified prior work on data-driven diagnostic cutoffs and repeated test-set reuse, narrowed the novelty claim accordingly, and ordered both bibliographies by first citation. These literature and reference-order updates added no model, cohort, or performance result.

The v88 snapshot includes the current IEEE and arXiv manuscript sources, with an aggregate ROC figure and a regression check tying its plotted coordinates and operating-point markers to the hash-verified run card. It also states that the MMRDR referable score includes a grade-5 logit despite no grade-5 categorical labels; possible indirect binary/ranking supervision is distinguished from absent direct grade supervision, and the unavailable checkpoint leaves the metric effect unknown. It foregrounds the primary empirical contrast: post-hoc image-level ROC-AUC 0.9173 coexists with 20.02% specificity at the frozen high-sensitivity operating point. At >=99% sensitivity, the best test-derived specificity was 30.94%; at 100% observed sensitivity it was 9.40%. These diagnostics use inspected test labels and are strictly descriptive. Prior literature summaries remain contextual, not a comparator; source-mixture reweighting remains arithmetic on exposed outcomes, not a model gain. Fresh-extraction checks and aggregate-audit limits are recorded in the receipt. These checks do not reproduce MMRDR inference or the private row-level score replay. The evaluated checkpoint and exact run-specific launch arguments are unavailable. The archive excludes images, patient/per-eye rows, private predictions, score/label replay inputs, feature caches, and checkpoints.

The IEEE source compiles in the desktop editor, but that compiler does not export a PDF. No current PDF built with the unmodified publisher class has been visually inspected. Author declarations, source-specific rights review, full inference reproducibility, and final author approval remain unresolved; this is a review package, not a cleared journal submission.

# One-Hour Paper Sprint Status

Status: IEEE-format draft generated and compiled. The claim has been upgraded
from image-count savings alone to an operational cost-risk-resolution frontier.

## Generated Artifacts

- `paper/active_retina_rc_fast_draft.tex`: compiled IEEEtran manuscript draft.
- `results/fast_paper_evidence_pack.json`: reproducible evidence pack for the
  draft tables and decision.
- `results/table_fast_paired_value.csv`: paired-view AUROC/AUPRC/Brier table.
- `results/table_fast_policy_baselines.csv`: adaptive policy versus baseline
  table.
- `src/active_retina/write_fast_paper_pack.py`: generator for the above
  artifacts.

## Current Evidence

- DeepDRiD paired views improve validation AUROC under both suffix branches:
  +0.041 for the suffix2-initial branch and +0.020 for the suffix1-initial
  branch.
- The legacy adaptive policy reduces mean image use by about 40% versus
  always-two-view acquisition in development validation.
- The upgraded two-stage policy allows the acquired second view to make either
  a negative or positive automatic decision. It eliminates validation deferrals
  at the same observed missed-referable-DR rate as confidence triage while still
  saving 41-46% of images versus always-two-view acquisition.
- Relative to first-view confidence triage, the two-stage policy uses 0.077 to
  0.179 more images per eye but removes 0.064 to 0.128 deferred cases per eye.
  This is favorable whenever one deferred case costs more than about 1.2-1.4
  image-equivalent units operationally.
- Risk control is empirical and uncertified. No final test claim is made because
  the historical test split was exposed during method repair.

## Decision

GO-ARXIV / CONDITIONAL-IEEE.

The paper is now viable as an arXiv methods flagship if presented as development
evidence. For an IEEE journal claim, the two-stage policy must be frozen and
confirmed on MFIDDR or another independent paired/multi-field cohort with formal
risk bounds computed before final test inspection.

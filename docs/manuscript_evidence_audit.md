# Manuscript Evidence Audit

The reproducible audit is `python -m active_retina.audit_baselines` from the project root with `PYTHONPATH=src`. It reads development predictions and previously selected policies only. It verifies disjoint patient IDs across training, calibration, and validation, accepted prediction scopes, and absence of test rows. It records prediction SHA-256 hashes. Upstream model provenance still requires independent verification; metadata assertions alone are insufficient.

## Corrections Implemented

- Table II conservative policies use calibration target 0.01. The earlier Table III mixed target 0.05 policies and a different two-stage algorithm. They were not the same operating point.
- Blank fixed-baseline rows represented calibration infeasibility, not missing evaluation results. Their risk-constrained fallbacks are now evaluated and explicitly marked infeasible under the joint constraints.
- Shared-target confidence triage and acquisition are compared at targets 0.01, 0.05, and 0.10 with minimum calibration specificity 0.50.
- Paired patient-bootstrap uncertainty is computed from 1,000 shared resamples, seed 20261003. Thresholds and models remain fixed. Validation was already inspected; no new confirmation is claimed.
- Permutation-invariant mean/max/cardinality aggregation and classifier settings are now documented accurately.
- Conditional miss risk, specificity, and coverage are defined. No minimum calibration coverage was enforced historically; low conditional risk cannot be interpreted without deferral reporting.

## Findings

At calibration target 0.01, acquisition uses 1.205/1.410 images and defers 20.5%/39.7%; confidence triage uses one image and defers 7.7%/19.2%. Acquisition has lower observed miss risk (3.6%/0% versus 6.3%/6.5%). This is a tradeoff, not demonstrated superiority. At unit deferral cost, acquisition costs more in both branches, with paired-bootstrap differences above zero. At target 0.10 and deferral cost 10, the apparent acquisition cost improvement has intervals crossing zero.

Paired discrimination gains and conservative policy counts remain supported by existing development outputs. The zero-event bootstrap interval is degenerate and cannot certify population risk. Five-class predictions remain unavailable. Anatomy mapping remains unofficial. Historical test exposure is irreversible. New patient splits drawn from already inspected patients cannot restore an untouched test cohort.

## Research Argument and Closure

The contribution to test is a risk--coverage--cost frontier improvement over strong deferral and fixed-acquisition comparators. Pre-register risk, specificity, minimum coverage, cost ranges, patient clustering, policy-search budget, and selection rules before a fresh cohort is evaluated. Include false-positive and missed-disease costs before clinical cost claims. A definitive superiority claim requires independent paired-view confirmation, replicated training uncertainty, and complete secondary grading. Single-image robustness can support classifier transportability but cannot confirm acquisition benefit.

Current decision: acquisition superiority unestablished; continue research. Manuscript formatting does not change this decision.

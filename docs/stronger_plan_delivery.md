# Development draft correction

Decision: NOT READY FOR ARXIV OR IEEE SUBMISSION. The previous GO-IEEE-candidate label was too strong and has been retracted.

The IEEEtran manuscript is `paper/active_retina_rc_fast_draft.tex`. It is an internal development draft. It reports a conservative risk-first selective second-view policy as the main development result, plus the zero-deferral policy as an operational ablation. It is not ready for public preprint release or journal submission.

New reproducible evidence:
- `results/stronger_plan_uncertainty.json`: 1,000 paired patient-cluster bootstrap replicates, seed 20261003, frozen thresholds, source hashes.
- `results/table_operational_cost_sensitivity.csv`: incremental image-plus-deferral costs with paired intervals across explicit hypothetical deferral costs.
- `results/ieee_gate_decision.json`: corrected readiness decision and required blockers.
- `results/table_ieee_gate_policy.csv`: conservative target-risk policy table used in the revised manuscript.
- Reproduce the added analyses with `PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.strengthen_paper` and `PYTHONPATH=src .venv-active-retina/bin/python -m active_retina.ieee_gate_upgrade` from the project root. Run the fast pack generator first only when intentionally rebuilding its source outputs; it replaces manuscript text.

Validation contains 39 patients / 78 eyes. The conservative policy misses 1/28 decided referable eyes (3.6%) in suffix2_initial and 0/27 decided referable eyes (0.0%) in suffix1_initial, with specificity 0.559 and 0.550. It saves 29% to 40% of images versus fixed two-view acquisition, while deferring 20.5% to 39.7% of eyes. Bootstrap missed-eye-risk intervals remain wide; no certified safety, equivalence, noninferiority, or clinical cost-effectiveness claim is justified.

Submission blockers: clean unexposed test or external paired validation, broader novelty review, stronger acquisition and selective-classification comparators, appropriate risk bounds, full training/selection uncertainty, complete five-class secondary analysis, stable handling of the unresolved suffix-to-anatomy mapping, detailed model and protocol reporting, clinical workflow and cost framing, and broad literature coverage. Readiness should be judged by evidence depth and study completeness, not by a nominal page count. Historical test exposure remains disclosed. External single-image data cannot replace paired acquisition confirmation. No CCG or four-field planning claim is made.

The manuscript has been corrected back to an honest development report. The next milestone is evidence generation and manuscript expansion, not submission.

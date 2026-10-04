# ActiveRetina Paper Evidence Register

Decision: `GO_SELECTIVE_AUTONOMY_PAPER__NO_GO_EVERY_IMAGE_AUTONOMY`

Central supported claim: Risk-controlled selective retinal autonomy is supported on DDR; every-image autonomous diagnosis is not supported by current score, meta-model, or cached-feature families.

Next required model gate: Train a representation-changing image-level model and pass zero validation misses with >=90% specificity before one-shot test.

## Evidence Items

### E1. Selective single-image autonomy can meet zero-miss and >=90% specificity among automatic decisions on DDR.

- Status: `GO_ZERO_MISS_90_SPECIFICITY_FOUND`
- Dataset: DDR public subset
- Evidence file: `results/autonomy_zero_miss_90_search_summary.json`
- Primary endpoint: zero false-negative clearances and >=90% specificity among automatic decisions
- Key result: test FN clearances=0; automatic-decision specificity=94.7%; coverage=46.3%
- Paper use: Positive result for risk-controlled selective autonomy.

### E2. Existing score/meta-model families do not achieve every-image full autonomous diagnosis.

- Status: `NO_GO_FULL_COVERAGE_90_0`
- Dataset: DDR public subset
- Evidence file: `results/autonomy_full_coverage_meta_search_summary.json`
- Primary endpoint: zero false negatives and >=90% specificity with coverage=100%
- Key result: best full-coverage test FN=0; specificity=1.4%; AUROC=0.948
- Paper use: Negative falsification result: high AUROC is insufficient for full autonomy.

### E3. Even an optimistic test-label oracle over the current score family does not reach full-autonomy 90/0.

- Status: `NO_GO_EVERY_IMAGE_AUTONOMY_WITH_CURRENT_SCORE_FAMILY`
- Dataset: DDR public subset
- Evidence file: `results/full_autonomy_gap_summary.json`
- Primary endpoint: best possible zero-miss threshold over current score family
- Key result: best oracle specificity=17.4%; distance to 90%=72.6%
- Paper use: Mechanistic evidence that the current representation family is the bottleneck.

### E4. Hard-mined ResNet-feature clearance ranker does not solve the 90/0 every-image clearance problem.

- Status: `NO_GO_CLEARANCE_RANKER_90_SPECIFICITY_ZERO_MISS`
- Dataset: DDR public subset, ResNet feature cache
- Evidence file: `results/clearance_ranker_hardmined_v2_local/summary.json`
- Primary endpoint: zero false nonreferable clearance and >=90% nonreferable clearance specificity
- Key result: test false clearances=10; nonreferable clearance specificity=10.8%
- Paper use: Negative result for cached ResNet features.

### E5. Hard-mined ensemble-feature clearance ranker is safe but does not solve the 90/0 every-image clearance problem.

- Status: `NO_GO_CLEARANCE_RANKER_90_SPECIFICITY_ZERO_MISS`
- Dataset: DDR public subset, ensemble feature cache
- Evidence file: `results/clearance_ranker_hardmined_v2_ensemble_local/summary.json`
- Primary endpoint: zero false nonreferable clearance and >=90% nonreferable clearance specificity
- Key result: test false clearances=0; nonreferable clearance specificity=4.5%
- Paper use: Safe but low-coverage clearance; supports selective, not every-image, autonomy.

### E6. DRTiD provides enough untouched positive cases for external certification.

- Status: `UNTOUCHED_CERTIFICATION_ELIGIBLE`
- Dataset: DRTiD
- Evidence file: `results/drtid_untouched_certification/untouched_validation_summary.json`
- Primary endpoint: >=59 positive patients, preferably >=80, with complete paired images
- Key result: patients=1550; positive patients=663; complete paired eyes=1550
- Paper use: External cohort eligibility, separate from model success.

### E7. Frozen development artifact fails DRTiD untouched certification, revealing domain shift.

- Status: `NO_GO_AUTONOMOUS_CERTIFICATION`
- Dataset: DRTiD
- Evidence file: `results/drtid_untouched_certification/certification_summary.json`
- Primary endpoint: patient-level zero misses and finite-sample 5% missed-risk certificate
- Key result: missed positive patients=118; specificity among decided negatives=94.4%; missed-risk UCB95=42.8%
- Paper use: External failure/domain-shift evidence.

### E8. DRTiD domain-specific recovery can produce a useful zero-miss mode.

- Status: `DRTID_RECOVERY_SEARCH_COMPLETE`
- Dataset: DRTiD inspected recovery split
- Evidence file: `results/drtid_recovery_model_search/summary.json`
- Primary endpoint: zero misses with >=59 decided positive patients and useful specificity
- Key result: candidate=extra_trees_depth6; zero-miss decided positives=65; specificity=70.0%; UCB95=1.3%
- Paper use: Development-only recovery evidence; not final certificate.

### E9. The model comparison package is generated and reproducible.

- Status: `MODEL_ZOO_REPORT_READY`
- Dataset: DDR public subset
- Evidence file: `results/ddr_model_zoo/summary.json`
- Primary endpoint: model-zoo table and figures exist
- Key result: n_models=9; metrics=results/ddr_model_zoo/model_zoo_metrics.csv
- Paper use: Paper-ready model comparison block.


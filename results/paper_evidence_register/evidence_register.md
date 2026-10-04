# ActiveRetina Paper Evidence Register

Decision: `GO_AUGMENTED_THREE_OUTPUT_AUTONOMOUS_TRIAGE__NO_GO_EVERY_IMAGE_BINARY_DIAGNOSIS`

Central supported claim: Every-input autonomous triage is supported on DDR when insufficient-evidence/reacquire is an explicit output. Adding a public EfficientNet fundus signal improves the diagnostic coverage frontier from 43.8% to 66.7% while preserving zero false-negative clearances and meeting >=90% diagnostic specificity. Every-image binary autonomous diagnosis remains unsupported by current score, meta-model, or cached-feature families.

Next required model gate: Train and freeze an explicit learned fusion model over the augmented signal family, then pass zero validation misses with >=90% specificity before one-shot test or external certification.

## Evidence Items

### E1. Every input can receive a completed autonomous product output using a three-output triage contract.

- Status: `GO_THREE_OUTPUT_AUTONOMOUS_TRIAGE`
- Dataset: DDR public subset
- Evidence file: `results/three_output_autonomous_triage/summary.json`
- Primary endpoint: autonomous output for every image, zero false-negative clearances, and >=90% specificity among diagnostic decisions
- Key result: autonomous output rate=100.0%; false-negative clearances=0; diagnostic-decision specificity=94.7%; diagnostic decision rate=43.8%; insufficient-evidence rate=56.2%
- Paper use: Positive product-output result; insufficient evidence is a completed autonomous workflow action, not a binary diagnosis.

### E2. Selective single-image autonomy can meet zero-miss and >=90% specificity among automatic decisions on DDR.

- Status: `GO_ZERO_MISS_90_SPECIFICITY_FOUND`
- Dataset: DDR public subset
- Evidence file: `results/autonomy_zero_miss_90_search_summary.json`
- Primary endpoint: zero false-negative clearances and >=90% specificity among automatic decisions
- Key result: test FN clearances=0; automatic-decision specificity=94.7%; coverage=46.3%
- Paper use: Positive result for risk-controlled selective autonomy.

### E3. Existing score/meta-model families do not achieve every-image full autonomous diagnosis.

- Status: `NO_GO_FULL_COVERAGE_90_0`
- Dataset: DDR public subset
- Evidence file: `results/autonomy_full_coverage_meta_search_summary.json`
- Primary endpoint: zero false negatives and >=90% specificity with coverage=100%
- Key result: best full-coverage test FN=0; specificity=1.4%; AUROC=0.948
- Paper use: Negative falsification result: high AUROC is insufficient for full autonomy.

### E4. Even an optimistic test-label oracle over the current score family does not reach full-autonomy 90/0.

- Status: `NO_GO_EVERY_IMAGE_AUTONOMY_WITH_CURRENT_SCORE_FAMILY`
- Dataset: DDR public subset
- Evidence file: `results/full_autonomy_gap_summary.json`
- Primary endpoint: best possible zero-miss threshold over current score family
- Key result: best oracle specificity=17.4%; distance to 90%=72.6%
- Paper use: Mechanistic evidence that the current representation family is the bottleneck.

### E5. Hard-mined ResNet-feature clearance ranker does not solve the 90/0 every-image clearance problem.

- Status: `NO_GO_CLEARANCE_RANKER_90_SPECIFICITY_ZERO_MISS`
- Dataset: DDR public subset, ResNet feature cache
- Evidence file: `results/clearance_ranker_hardmined_v2_local/summary.json`
- Primary endpoint: zero false nonreferable clearance and >=90% nonreferable clearance specificity
- Key result: test false clearances=10; nonreferable clearance specificity=10.8%
- Paper use: Negative result for cached ResNet features.

### E6. Hard-mined ensemble-feature clearance ranker is safe but does not solve the 90/0 every-image clearance problem.

- Status: `NO_GO_CLEARANCE_RANKER_90_SPECIFICITY_ZERO_MISS`
- Dataset: DDR public subset, ensemble feature cache
- Evidence file: `results/clearance_ranker_hardmined_v2_ensemble_local/summary.json`
- Primary endpoint: zero false nonreferable clearance and >=90% nonreferable clearance specificity
- Key result: test false clearances=0; nonreferable clearance specificity=4.5%
- Paper use: Safe but low-coverage clearance; supports selective, not every-image, autonomy.

### E7. DRTiD provides enough untouched positive cases for external certification.

- Status: `UNTOUCHED_CERTIFICATION_ELIGIBLE`
- Dataset: DRTiD
- Evidence file: `results/drtid_untouched_certification/untouched_validation_summary.json`
- Primary endpoint: >=59 positive patients, preferably >=80, with complete paired images
- Key result: patients=1550; positive patients=663; complete paired eyes=1550
- Paper use: External cohort eligibility, separate from model success.

### E8. Frozen development artifact fails DRTiD untouched certification, revealing domain shift.

- Status: `NO_GO_AUTONOMOUS_CERTIFICATION`
- Dataset: DRTiD
- Evidence file: `results/drtid_untouched_certification/certification_summary.json`
- Primary endpoint: patient-level zero misses and finite-sample 5% missed-risk certificate
- Key result: missed positive patients=118; specificity among decided negatives=94.4%; missed-risk UCB95=42.8%
- Paper use: External failure/domain-shift evidence.

### E9. DRTiD domain-specific recovery can produce a useful zero-miss mode.

- Status: `DRTID_RECOVERY_SEARCH_COMPLETE`
- Dataset: DRTiD inspected recovery split
- Evidence file: `results/drtid_recovery_model_search/summary.json`
- Primary endpoint: zero misses with >=59 decided positive patients and useful specificity
- Key result: candidate=extra_trees_depth6; zero-miss decided positives=65; specificity=70.0%; UCB95=1.3%
- Paper use: Development-only recovery evidence; not final certificate.

### E10. The model comparison package is generated and reproducible.

- Status: `MODEL_ZOO_REPORT_READY`
- Dataset: DDR public subset
- Evidence file: `results/ddr_model_zoo/summary.json`
- Primary endpoint: model-zoo table and figures exist
- Key result: n_models=9; metrics=results/ddr_model_zoo/model_zoo_metrics.csv
- Paper use: Paper-ready model comparison block.

### E11. Public EfficientNet fundus signal improves the autonomous triage frontier when added to the ensemble.

- Status: `GO_EXTRA_SIGNAL_IMPROVES_AUTONOMY`
- Dataset: DDR public subset
- Evidence files: `results/autonomy_extra_signal_clementp_efficientnet_b5_random2000_fast/summary.json`; `results/autonomy_extra_signal_clementp_efficientnet_b5_random600_spec925/summary.json`
- Primary endpoint: improved diagnostic decision rate while preserving zero false-negative clearances and >=90% specificity among diagnostic decisions
- Key result: max-coverage mode test FN clearances=0; diagnostic-decision specificity=90.3%; diagnostic decision rate=66.7%; insufficient-evidence rate=33.3%; negative-clearance rate=38.0%
- Conservative mode: test FN clearances=0; diagnostic-decision specificity=93.1%; diagnostic decision rate=54.4%; insufficient-evidence rate=45.6%; negative-clearance rate=27.6%
- Paper use: Positive model-improvement result; shows the autonomy frontier can be improved with external retinal pretrained signal diversity.

### E12. Learned fusion over augmented signals improves coverage but fails the safety gate.

- Status: `NO_GO_AUGMENTED_LEARNED_FUSION`
- Dataset: DDR public subset
- Evidence files: `results/autonomy_augmented_learned_fusion_clementp_efficientnet_b5/summary.json`; `results/autonomy_augmented_learned_fusion_clementp_efficientnet_b5_spec99/summary.json`
- Primary endpoint: learned fusion trained on inner validation-train, calibrated on held-out validation-calibration, and reported once on test
- Key result: best high-specificity learned mode reached 52.5% diagnostic decision rate and 98.1% specificity, but introduced 2 false-negative clearances
- Paper use: Negative model-selection evidence; learned fusion is not accepted despite attractive coverage/specificity because zero-miss safety is mandatory.

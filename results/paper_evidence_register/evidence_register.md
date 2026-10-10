# ActiveRetina Paper Evidence Register

Decision: `DEVELOPMENT_ONLY_THREE_OUTPUT_FRONTIER__NO_GO_EVERY_IMAGE_BINARY_DIAGNOSIS`

Central supported claim: Existing score signals support only exploratory three-output triage on the inspected DDR cohort. The corrected validation-only score-mixture result is 83.0% specificity among 72.6% diagnostic decisions with zero observed misses on an exposed test split; forcing deferred cases to referral yields 33.9% binary specificity. Every-image binary autonomous diagnosis remains unsupported.

Next required model gate: Complete the paired-view residual experiment, selected and thresholded on paired validation data only. Further model work must show a material validation gain without reusing test results for candidate choice.

## Evidence Items

### E1. Every input can receive a completed autonomous product output using a three-output triage contract.

- Status: `GO_THREE_OUTPUT_AUTONOMOUS_TRIAGE`
- Dataset: DDR public subset
- Evidence file: `results/three_output_autonomous_triage/summary.json`
- Primary endpoint: autonomous output for every image, zero false-negative clearances, and >=90% specificity among diagnostic decisions
- Key result: autonomous output rate=100.0%; false-negative clearances=0; diagnostic-decision specificity=94.7%; diagnostic decision rate=43.8%; insufficient-evidence rate=56.2%
- Paper use: Positive product-output result; insufficient evidence is a completed autonomous workflow action, not a binary diagnosis.

### E2. The earlier selective-search GO was test-selected and is superseded.

- Status: `DEVELOPMENT_ONLY_TEST_EXPOSED`
- Dataset: DDR public subset
- Evidence file: `results/autonomy_zero_miss_90_search_summary.json`
- Primary endpoint: zero false-negative clearances and >=90% specificity among automatic decisions
- Key result: corrected validation-only selection chose `mlp_ensemble`; exposed-test diagnostic had 2 false-negative clearances, 83.3% specificity among decisions, and 54.6% diagnostic coverage.
- Paper use: Development diagnostic only; the historical 94.7% / 46.3% test-selected headline is not confirmatory evidence.

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

### E11. Validation-selected augmented scores define a development triage frontier, not a GO.

- Status: `DEVELOPMENT_ONLY_TEST_EXPOSED`
- Dataset: DDR public subset
- Evidence files: `results/autonomy_extra_signal_clementp_efficientnet_b5_random5000_fast/summary.json`; `results/autonomy_extra_signal_clementp_efficientnet_b5_random600_spec925/summary.json`
- Primary endpoint: improved diagnostic decision rate while preserving zero false-negative clearances and >=90% specificity among diagnostic decisions
- Key result: validation-only selection chose `random_weight_3992`; exposed-test diagnostic: 0 observed misses, 83.0% specificity among 72.6% diagnostic decisions, 27.4% insufficient-evidence rate. If insufficient cases are forced to REFER, binary specificity is 33.9%.
- Paper use: Exploratory frontier only. The earlier test-selected 90.8% / 66.8% headline is superseded.

### E12. Learned fusion over augmented signals improves coverage but fails the safety gate.

- Status: `NO_GO_AUGMENTED_LEARNED_FUSION`
- Dataset: DDR public subset
- Evidence files: `results/autonomy_augmented_learned_fusion_clementp_efficientnet_b5/summary.json`; `results/autonomy_augmented_learned_fusion_clementp_efficientnet_b5_spec99/summary.json`
- Primary endpoint: learned fusion trained on inner validation-train, calibrated on held-out validation-calibration, and reported once on test
- Key result: best high-specificity learned mode reached 52.5% diagnostic decision rate and 98.1% specificity, but introduced 2 false-negative clearances
- Paper use: Negative model-selection evidence; learned fusion is not accepted despite attractive coverage/specificity because zero-miss safety is mandatory.

### E13. Augmented score family still cannot support every-image binary autonomous diagnosis.

- Status: `DEVELOPMENT_ONLY_TEST_EXPOSED`
- Dataset: DDR public subset
- Evidence file: `results/augmented_full_binary_gap_validation_audit_20261011_clean/summary.json`
- Primary endpoint: every image receives a binary nonreferable/referable diagnosis with zero false negatives and >=80% specificity
- Key result: validation-selected candidate reached 44.1% validation specificity at zero misses; previously inspected test diagnostic was 33.1% specificity with 1 false negative.
- Paper use: Falsification result; the accepted product claim must remain three-output selective autonomy, not unrestricted binary diagnosis.

### E14. Validation-selected augmented candidate exposes the remaining uncertain-case burden.

- Status: `DEVELOPMENT_ONLY_TEST_EXPOSED`
- Dataset: DDR public subset
- Evidence files: `results/augmented_autonomy_gap_validation_audit_20261011/report.md`; local case lists are not for GitHub.
- Primary endpoint: characterize what still blocks every-image binary autonomy after the accepted augmented selective policy
- Key result: previously inspected test split abstentions=393/1435; abstained positives=104; abstained negatives=289; false-positive nonreferable referrals=34; false-negative clearances=0. If abstentions are forced to REFER, binary specificity is 166/489=33.9%.
- Paper use: Development mechanism analysis only; a real separate cohort is still required for confirmation.

# Source-Specialist Model Audit (2026-10-11)

## Decision

`NO_GO_80_0_100`. Selecting a checkpoint, score rule, and threshold separately for each dataset improved the development validation operating point, but the fixed choices missed referable cases on the already-exposed test split. This is a stability diagnostic, not independent confirmation.

## Protocol

Two existing RTX checkpoints were audited: the log-mean-exp multiple-instance candidate at 224 pixels and the source-balanced candidate at 384 pixels. For each checkpoint, five existing score rules were evaluated. Within each dataset source, the expert was selected using validation misses first and specificity second; the selected expert and its validation-selected threshold were applied unchanged to the test partition. No model was retrained and no test labels were used in selection.

The test partition has already been inspected in prior development. Every test metric below is therefore exposed-test diagnostic evidence only.

## Results

| Split | Gradable nonreferable | False referrals | Specificity | Referable DR | Misses | Case action |
|---|---:|---:|---:|---:|---:|---:|
| Validation, selected experts | 386 | 182 | 52.85% | 463 | 0 | 100% |
| Exposed test, fixed experts | 404 | 179 | 55.69% | 390 | 7 | 100% |

Known-ungradable cases were routed to referral with zero misses in both partitions. The 100% figure means every case received a binary action; it is not a claim that every action was clinically correct.

| Source | Validation expert | Validation specificity / misses | Exposed-test specificity / misses |
|---|---|---:|---:|
| DDR | Log-mean-exp checkpoint, ordinal score | 56.04% / 0 | 53.15% / 3 |
| DeepDRiD | Log-mean-exp checkpoint, ordinal score | 75.00% / 0 | 80.85% / 1 |
| DRTiD | Source-balanced checkpoint, mean of fused and single-view risks | 38.58% / 0 | 51.11% / 3 |

The aggregate validation specificity is 204/386; the exposed-test specificity is 225/404. The test result is not a candidate operating point because it has seven false-negative clearances and was used only to diagnose stability.

## Interpretation and Next Gate

This confirms modest development-set specificity recovery from model/score specialization, not full autonomy. The largest persistent error source remains the grade-1/grade-2 boundary, with severe source variation. Threshold and expert routing do not solve that representation/label-boundary problem, and source identity may not be available or stable in deployment.

Do not promote this policy to a product artifact or paper claim. The next model change should require a materially new training signal or a newly available paired-view cohort; repeating the same checkpoint/threshold search is not justified. Any later candidate needs patient-disjoint validation and a genuinely untouched, predeclared certification cohort.

Machine-readable outputs: `results/source_specialist_model_selection_20261011.json` and the two checkpoint audit summaries beside it.

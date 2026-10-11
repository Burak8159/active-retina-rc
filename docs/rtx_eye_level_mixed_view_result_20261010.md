# RTX Eye-Level Mixed-View Development Result

Status: `NO_GO_80_SPECIFICITY_ZERO_MISS`; development evidence only.

## Correction and method

The earlier merged run treated each DRTiD frame as an independently labeled case even though its grade is attached to a macula/optic-disc pair. The corrected trainer groups the two views into one eye-level input. DDR remains a single-image eye-level case; its grade-5 label is handled as ungradable rather than referable disease. The shared model is a pretrained `convnext_tiny.fb_in22k_ft_in1k` encoder with grade, referable-risk, and quality heads, mean/max/difference view fusion, cross-entropy/BCE objectives, and a hard-boundary ranking term.

The mixed manifest contains 4,238 single-view DDR cases and 1,550 paired DRTiD eyes. The initial run used 4,085 train eyes, 876 validation eyes, and 827 development-test eyes. It ran five epochs on the RTX 3050 Ti; the best validation checkpoint was epoch 2.

## Results at the validation-selected policy

| Split | Specificity on gradable grades 0-1 | Referable sensitivity, grades 2-4 | Missed referable eyes | Referral rate |
|---|---:|---:|---:|---:|
| Validation, 876 eyes | 31.1% (104/334) | 100% (409/409) | 0 | 88.1% |
| Development test, 827 eyes | 26.9% (96/357) | 99.4% (339/341) | 2 | 88.1% |

The development-test source breakdown is material: DDR specificity was 35.6% (79/222) with 100% sensitivity (259/259); DRTiD specificity was 12.6% (17/135) with 97.6% sensitivity (80/82). All 129 DDR grade-5 cases were routed to referral. DRTiD has no quality labels in this pipeline.

## Lower-learning-rate warm-start follow-up

A prespecified five-epoch refinement was initialized from the initial best checkpoint (epoch 2), using backbone learning rate `1e-6`, head learning rate `6e-5`, and a reduced hard-boundary ranking weight. The selected checkpoint was epoch 3, not the final epoch. Its validation operating point had specificity 30.2% (101/334), sensitivity 100% (409/409), zero missed referable eyes, and an 88.5% referral rate. On the development test, using that validation-selected policy, specificity was 29.4% (105/357), sensitivity 99.1% (338/341), and 3 referable eyes were missed; referral rate was 86.9%.

By source on the development test, DDR specificity was 36.0% (80/222) with sensitivity 99.6% (258/259), while DRTiD specificity was 18.5% (25/135) with sensitivity 97.6% (80/82). Relative to the initial mixed-view run, this is a small DRTiD specificity gain (+5.9 percentage points), but it does not improve the selected validation specificity and adds one development-test miss. The refinement is therefore not a success against the project target and is not evidence of clinical readiness. The test split remains development-only.

## Objective and score diagnostics

The separate risk head had development-test AUROC 0.960 on DDR and 0.868 on DRTiD, but the operational zero-validation-miss threshold still referred most grade-1 eyes. All three misses under the refined policy were grade 2: one DDR and two DRTiD. This points to overlap at the grade-1/grade-2 boundary and a remaining source shift, rather than a threshold-only issue.

Two bounded alternatives were checked without using the test split for selection:

- The existing five-class head's ordinal probability mass, `P(grade 2-4)`, reached 26.0% validation specificity with zero validation misses; its test specificity was 26.3% with 3 misses. It was inferior to the separate risk head.
- A validation-only sweep blending standardized risk-head and ordinal scores selected the risk head alone. No blend improved its 30.2% validation specificity.
- A cumulative ordinal objective (four ordered grade-threshold logits, additional grade-1/2 boundary BCE, and hard-boundary ranking) reached 29.0% validation specificity at epoch 1, then 21.6% and 22.8% at epochs 2-3, always with zero validation misses. It was stopped at epoch 3; no test evaluation was run for this family. An initial launch found an empty-mask NaN for all-ungradable batches; the losses were changed to use the existing masked-BCE helper and a regression test was added before retrying.

These experiments do not approach the requested 80% specificity / zero-miss / full-action target. The cumulative ordinal checkpoint is not promoted. Do not repeat head-only reweighting or threshold blending. A next candidate should change sampling/training exposure at the observed grade-1/grade-2 boundary and address cross-source robustness, with validation as the only selection gate.

`case_action_coverage=100%` means the program emits a binary `REFER` or `CLEAR` action for every eye. It does not mean 100% diagnostic coverage at target performance, and the 88.1% referral rate shows this policy is not clinically useful as a clearance system. Two development-test positives were missed, so it fails even the zero-miss requirement.

## Reproducibility artifacts

- Trainers and audits: `src/active_retina/train_mixed_view_bag_autonomy.py`, `src/active_retina/audit_mixed_view_ordinal_head.py`, and `src/active_retina/train_mixed_view_cumulative_ordinal.py`
- Regression checks: `tests/test_quality_gated_autonomy.py`
- RTX summaries and per-eye predictions: `results/rtx_mixed_view_bag_dev_20261010/`, `results/rtx_mixed_view_bag_refine_dev_20261010/`, and `results/ordinal_head_audit_dev_20261010/`
- Manifest: `results/autonomy_route_combined_manifest_resplit.csv`

The manifest and its partitions have already informed prior development. Neither validation nor test metrics here are an untouched certification result. Do not repeat this same warm-start family. Before another training run, audit per-grade and per-source score distributions and the missed-positive eyes, then choose a materially different, validation-only intervention. No threshold is to be selected on the development test.

## Follow-up experiments (2026-10-11)

### Boundary-focused sampling: stopped

A source-matched grade-1/grade-2 sampler was added to make the referral boundary explicit in most updates, with periodic grade-5 quality and grade-0 versus grade-3/4 control batches. Warm-starting the best mixed-view checkpoint and evaluating after epoch 1 yielded 23.65% validation specificity at zero referable misses, with a 91.0% referral rate. This failed the predeclared gate of improving on the 30.24% baseline, so training was stopped before using the development test. No checkpoint from this branch is promoted.

### Higher-resolution fine-tuning: stopped

A distinct 320-pixel, one-eye-batch candidate used source-balanced grade-boundary exposure and the same quality/extreme-grade controls. Epoch 1 reached 23.05% validation specificity at zero misses and a 91.2% referral rate. This also failed the 30.24% gate and was stopped without test evaluation. Increasing input resolution did not improve the current bottleneck in this short controlled attempt.

### Paired-view late-fusion audit

With the weights frozen, validation selected the original fused eye score over the alternatives. Every rule maintained zero validation misses, but none improved specificity:

| Validation-selected score | Validation specificity | Validation referral rate | Development-test specificity | Development-test missed referable eyes |
|---|---:|---:|---:|---:|
| Fused eye score | 30.24% | 88.47% | 29.41% | 3 |
| Mean of fused and two masked-view scores | 29.04% | 88.93% | 25.21% | 1 |
| Maximum of the two masked-view scores | 27.84% | 89.38% | 23.81% | 1 |

The masked-view rules modestly reduced misses on this already exposed development test only by referring more negatives; they are not a useful specificity improvement and are not promoted. Thresholds for all test figures were selected on validation.

### Newly included training domain

The local project also contains DeepDRiD regular-fundus images, which were absent from the preceding DDR+DRTiD training manifest. A three-source patient-level development manifest now combines DDR (4,238 image rows), paired DRTiD (3,100 rows), and paired DeepDRiD (1,584 rows; 792 complete eyes). Five incomplete DeepDRiD eye pairs were excluded. The manifest has 6,580 eyes total (4,675 train, 982 validation, 923 development test); validation and test remain exposed development evidence, not certification. The corrected counts include each DDR single-image case as one eye, rather than incorrectly halving all source rows.

The current bag fusion uses symmetric mean, maximum, and absolute-difference features, so swapping the two DeepDRiD views does not change its input representation. This permits paired training without asserting that suffix-to-anatomy mapping is officially verified. DeepDRiD's CC BY-SA 4.0 license and attribution requirements must be retained in any paper/model release; the dataset itself is not to be pushed to GitHub.

### Three-source candidate: source-balanced training stopped

The source-balanced sampler was tested as a materially different response to the known domain shift: batches rotate equally across DDR, DRTiD, and DeepDRiD, while preserving each source's natural grade mix and periodically pairing DDR quality controls. It warm-started the frozen best mixed-view checkpoint. The first epoch produced 24.61% validation specificity at zero missed referable eyes (95/386 gradable nonreferable eyes cleared; 291 false referrals), with 90.33% of all cases routed to referral. This is below the 30.24% prior validation gate, so training was stopped before any development-test evaluation. The epoch-1 checkpoint is retained for provenance but is not promoted. Every case received a binary software action, but the resulting 90.3% referral rate is not useful diagnostic coverage and does not meet 80/0/100.

A low-cost source-conditioned threshold diagnostic on the earlier DDR+DRTiD candidate also failed to close the gap. Per-source validation thresholds gave 34.7% pooled specificity at zero validation misses, versus 30.2% with one pooled threshold. Applied unchanged to the already-exposed development test, this reached about 31.7% pooled specificity and still missed three referable eyes. Dataset identity is not guaranteed to be known for future camera inputs, so this is a diagnostic result, not a deployable solution.

The validation error profile localizes the weakness: at the earlier candidate's selected policy, 92.2% of DDR grade-1 and 100% of DRTiD grade-1 eyes were referred; 80.6% of DRTiD grade-0 eyes were also referred. The source-balanced three-source run worsened pooled specificity, so equal source exposure alone is not the answer. The failure is a learned ranking/label-domain overlap problem at the early-grade boundary, not a threshold-only problem.

### Retinal-pretrained B5 fine-tuning: stopped

The cached MIT-licensed ClementP FundusDRGrading EfficientNet-B5 was fine-tuned end-to-end on the three-source training split at 320 pixels. Because the published model card reports prior training on DDR, policy selection and validation metrics were restricted to DRTiD and DeepDRiD (356 validation eyes); no test split was evaluated. With a newly initialized referable-risk head, epoch 1 achieved 0% specificity at zero misses and referred every case. Epoch 2 reached only 2.79% specificity (5/179 nonreferable eyes cleared), with 98.6% referral and zero validation misses. Training was stopped. This is a negative result for replacing the model's trained classifier with a random head, not evidence that the B5 encoder itself is ineffective.

### Current decision and next gate

The best existing candidate remains the 224-pixel fused eye model at 30.24% validation specificity with zero validation misses. The three-source source-balanced candidate and the first retinal-pretrained B5 fine-tune both failed their validation gates and were not test-evaluated. On the three-source validation set, 80% specificity at zero misses requires correctly clearing at least 309 of 386 gradable nonreferable eyes; the source-balanced run cleared 95. On the paired-source validation set, the B5 run cleared only 5 of 179. The next distinct implementation is a residual adaptation that preserves the pretrained B5 disease logit and learns a paired-view correction around it; this directly avoids discarding the pretrained classifier as the failed run did. If that does not make a material validation gain, the currently available data/models have reached a pre-mBRSET ceiling and further threshold/sampler searches are not justified. The stated 80% specificity / zero-miss target remains unachieved; 100% action emission is operational completeness, not diagnostic autonomy.

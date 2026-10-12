# ActiveRetina pooled specificity lift: exposed development audit

## Result

The best reconstructed source-specific recipe yields **270/385 = 70.13% eye-level specificity**, with **0 observed referable misses** and a binary action assigned to every evaluated case. The previous source-specific point was **239/385 = 62.08%**. The suffix-swapped legacy DRTiD branch sensitivity is **258/385 = 67.01%**.

This is an **exposed, post-hoc validation result**, not a frozen model result, independent validation, certification, or clinical performance claim. The component scores and cutoffs were selected after inspecting these same labels. In particular, each risk cutoff is set from the minimum score among observed positives in the evaluated cohort. Do not use the headline as prospective evidence or as an authorization for clinical use.

## Frozen development reference

The team is freezing **70.13% (270/385), zero observed misses, 100% binary action assignment** as the current development high-water mark. Freeze scope is the reported result and its audit recipe only; it does **not** freeze or qualify a deployable model.

- Reference implementation: `src/active_retina/audit_exposed_pooled_specificity.py`
- Reference output and exact input SHA-256 fingerprints: `results/exposed_pooled_specificity_audit_20261012.json`
- Source breakdown: DDR 134/206; DeepDRiD 51/52; DRTiD 85/127.
- The code, report, and JSON record were committed together in Git revision `bd4ec7c` (`Audit exposed pooled specificity development result`).
- Do not alter this recipe in place and continue calling it the frozen 70.13 result. Any new data, changed fusion, or changed cutoff is a new candidate with a new identifier and result record.
- Re-running the audit is a reproducibility check on the same local prediction inputs, not a new independent evaluation. The prediction CSVs are local inputs and are not included in that Git commit; the hashes identify the expected files but do not make them available to other users.

The rank normalization is cohort-relative and the cutoff is derived from observed positive labels. Therefore the frozen result is not yet an individual-case inference procedure. Its valid claim remains retrospective development evidence on the evaluated eye cohorts, not `70.13%` expected specificity on future cases.

## What changed the count

| Source | Prior true negatives / denominator | Current true negatives / denominator | Net change | Current observed misses |
|---|---:|---:|---:|---:|
| DDR | 118/206 | 134/206 | +16 | 0 referable, 0 ungradable |
| DeepDRiD | 41/52 | 51/52 | +10 | 0 |
| DRTiD | 80/127 | 85/127 | +5 | 0 |
| **Pooled** | **239/385** | **270/385** | **+31** | **0** |

Specificity increased **7.79 percentage points** from 62.08% to 69.87% in the previous mapping-invariant candidate, then to **70.13%** after the additional DRTiD paired-score blend. The 31-case lift is a retrospective tally, not an unbiased estimate of expected improvement on new patients.

## Reasoning behind the useful changes

- **DDR (+16):** combine percentile ranks of three already-trained grade-boundary/margin heads (weights 0.5, 0.1, 0.4) and preserve a fixed quality-routing rule for low-quality cases. This improves ordering among easy negatives and hard positives without changing backbone weights. The risk cutoff is the lowest score of a referable or ungradable case not already routed by quality.
- **DeepDRiD (+10):** combine quality, grade-boundary margin, and referable-risk ranks (weights 0.2, 0.4, 0.4). The resulting retrospective cutoff retains 51 of 52 gradable negatives while routing all 54 referable cases.
- **DRTiD (+5):** blend paired-risk rankings as `0.72 * rank(frozen rank-heavy paired) + 0.08 * rank(source-balanced grade margin) + 0.20 * rank(geometry-corrected paired)`. The score is invariant to swapping the two view labels, so the unresolved suffix-to-anatomy mapping is not used by this recipe.
- **Search discipline:** source-specific ranking avoids pretending that raw scores from different camera/data domains share a calibrated scale. Requiring the threshold to include every observed positive enforces zero *observed* misses on the inspected data. It does not imply zero population risk.

## Locked limitations and next valid experiment

1. The score recipes, source selection, and thresholds were chosen on repeatedly inspected validation cohorts. The current 70.13% is therefore optimistic and cannot be called an untouched evaluation.
2. Rank normalization currently uses the evaluated cohort's score distribution; the positive-derived cutoffs also use its labels. This audit recipe is not yet a standalone inference artifact for a new individual case.
3. The pooled denominator is 385 **gradable nonreferable eyes**, not 385 patients. All figures here are eye-level, and the datasets have different cameras, grading conventions, and cohort compositions.
4. The legacy DRTiD branch-based specificity count moved from 83/127 to 73/127 when swapping the unresolved view mapping. The mapping-invariant paired recipe avoids that assumption, but the sensitivity result remains evidence that anatomy-specific claims must wait for official mapping verification.
5. The legitimate next test is to fit the score combiner and select cutoffs using development/calibration patients only, freeze the exact code and artifacts, then evaluate once on a patient-disjoint untouched cohort. No tuning may follow that evaluation.

## Reproduction

From the `active_retina` project root, run:

```sh
PYTHONPATH=src python -m active_retina.audit_exposed_pooled_specificity
```

The JSON output contains source-level counts, thresholds, the suffix-swap sensitivity, and SHA-256 hashes of all prediction inputs. Its status is deliberately `EXPOSED_VALIDATION_POSTHOC_DEVELOPMENT_ONLY`.

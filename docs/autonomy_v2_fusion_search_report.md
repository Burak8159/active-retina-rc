# ActiveRetina-Autonomy-v2 Fusion Search

Evaluated 30 score-fusion candidates using existing DDR predictions. Thresholds were selected on validation and reported on test.

## Best Forced Binary

`median_all_ranknorm` at target `0.999`: test sensitivity 100.0%, specificity 15.5%, false negatives 0.

## Best Selective

`top2_mean_all_ranknorm`: test coverage 67.5%, sensitivity 98.7%, auto-specificity 92.5%, false negatives 12.

## Decision

**NO_GO_90_SPECIFICITY_ZERO_MISS**. Score-level fusion alone is not enough if the reported best row still misses the target. The next model must train the fusion and active-acquisition policy directly, rather than only combining already-frozen probabilities.

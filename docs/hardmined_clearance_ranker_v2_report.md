# Hard-Mined Clearance Ranker V2

## Purpose

This experiment changes the model objective from generic referable-DR classification to the autonomous clearance failure mode:

> clear as many true nonreferable images as possible while clearing zero referable images.

The implementation extends `src/active_retina/train_clearance_ranker.py` with:

- asymmetric focal BCE;
- extra weight on grade-2 referable boundary cases;
- extra pressure on grade-1 nonreferable boundary cases;
- pairwise ranking loss between referable and nonreferable images;
- periodic hard-case mining, where low-scoring referable images and high-scoring nonreferable images are oversampled during training;
- validation-only zero-miss threshold selection followed by a single test evaluation.

## Runs

### ResNet-50 DDR feature cache

Command output:

- output: `results/clearance_ranker_hardmined_v2_local/summary.json`
- feature cache: `results/ddr_hf_specificity300_features.npz`
- validation: zero false clearances, 17.1% nonreferable clearance specificity
- test: 10 false clearances, 10.8% nonreferable clearance specificity
- status: `NO_GO_CLEARANCE_RANKER_90_SPECIFICITY_ZERO_MISS`

### Ensemble DDR feature cache

Command output:

- output: `results/clearance_ranker_hardmined_v2_ensemble_local/summary.json`
- feature cache: `results/ddr_hf_specificity100_ensemble_features.npz`
- validation: zero false clearances, 7.0% nonreferable clearance specificity
- test: zero false clearances, 4.5% nonreferable clearance specificity
- status: `NO_GO_CLEARANCE_RANKER_90_SPECIFICITY_ZERO_MISS`

## Interpretation

The hard-mined objective makes the model safer, but it does not solve every-image autonomous diagnosis. The ensemble feature run keeps zero test misses, but it only clears 9/200 nonreferable test images. The ResNet feature run clears more negatives, but it leaks 10 referable images on test.

This means the remaining blocker is not thresholding. The negative and positive tails overlap too much in the current cached feature space. The next credible route is an image-level model that changes the representation itself, not another head on the same frozen embeddings.

## Next Model Step

The next non-redundant experiment should be an end-to-end retinal model with:

- high-resolution input;
- lesion-aware or patch/token attention;
- explicit image-quality branch;
- ordinal DR supervision;
- hard-positive mining focused on grade-2 images;
- clearance-specific validation objective.

The project should not spend more time on cached-feature threshold searches for the every-image 90/0 target unless new features or new data are introduced.

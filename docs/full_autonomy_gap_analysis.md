# Full Autonomy Gap Analysis

Target: every test image receives CLEAR or REFER, with zero referable cases cleared and at least 90% specificity.

Status: **NO_GO_EVERY_IMAGE_AUTONOMY_WITH_CURRENT_SCORE_FAMILY**

Best optimistic oracle candidate: `median_all`

Best optimistic zero-miss specificity: **17.4%**

Distance to 90% specificity: **72.6 percentage points**

Current selective policy abstains on **771 / 1435** test images.

Abstained referable images: **358**; abstained nonreferable images: **413**.

Interpretation: the existing frozen score family does not contain enough information to safely force every case into an autonomous decision. The next valid fix is image-level hard-case training or additional view/quality evidence, not another threshold search.

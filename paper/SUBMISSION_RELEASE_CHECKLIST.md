# Submission and release checklist

Status is based on the repository and primary dataset-source pages checked on 2026-10-15. This is an author review checklist, not legal advice or institutional approval.

## Evidence and reproducibility

- [x] Re-execute the frozen policy on all 981 locally available score rows and compare every action and risk score.
- [x] Verify the replay source counts, 270/385 specificity numerator/denominator, zero replay clearances of referable or ungradable cases, and principal grouped/nested/transfer summary counts.
- [x] Emit input SHA-256 digests and a machine-readable audit report.
- [ ] Reproduction from a clean public clone: blocked because the score rows and policy are local/untracked pending the rights review below.
- [ ] Reproduce component scores from raw images: blocked; not all checkpoints, preprocessing, and inference paths are packaged.
- [ ] Re-run every figure/table from raw predictions: incomplete; current auditor verifies headline values and exact replay, not every manuscript cell or plot.
- [ ] Create an immutable release commit and archive source, generated PDF, audit report, and environment record together.
- [ ] Confirm that the compiled PDF corresponds to the exact submitted source revision. The desktop compiler reports successful compilation but does not export a PDF artifact, so there is currently no PDF file to hash or archive against a commit.
- [ ] Decide whether the project code is to remain under the repository's current `All rights reserved` terms or be distributed under an explicit license. Do not describe the repository as open source while the current license is in force.

## Dataset rights and data availability

| Source | What the primary source says | Release decision |
|---|---|---|
| DDR/OIA-DDR | The official dataset repository states CC BY-NC-SA 4.0 for images, annotations, labels, and dataset materials; its README also says commercial use needs written permission. [Official repository](https://github.com/nkicsl/DDR-dataset) | Cite and link to original source; do not place data or derived per-eye score/label rows in this public repo until the author confirms the license implications and noncommercial scope. |
| DeepDRiD | The official repository provides a CC BY-SA 4.0 `LICENSE` and describes research use. [Official repository](https://github.com/deepdrdoc/DeepDRiD) | Confirm that the repository license applies to the dataset files used and determine whether derived row-level outputs/policy references may be redistributed under compatible terms. |
| DRTiD | The official repository says access is through an application form, describes the dataset as research-only, and requires citation; it does not display a separate machine-readable dataset license. [Official repository](https://github.com/FDU-VTS/DRTiD) | Confirm the grant/access terms permit publication and public redistribution of derived eye-level scores, labels, and policy artifacts. Until then, keep these files out of GitHub. |
| MMRDR | The data descriptor points to a Figshare record. [Data descriptor](https://www.nature.com/articles/s41597-026-07005-9) | The precise item-level license and derivative-output terms were not verified from the primary Figshare record in this review. Verify record metadata and any restrictions before release. |
| IDRiD | The manuscript records use in a later training experiment, but the applicable image/data license and redistribution permission are not established in this package. | Confirm the exact release terms and whether training-derived artifacts are distributable. |

These entries record source statements, not a legal determination that sharing model outputs is or is not permitted. Public availability of images does not by itself resolve whether row-level predictions, labels, identifiers, or dataset-derived rank references may be republished. The paper's data statement must not promise open derived data until the named author documents the answer. No raw dataset images are included here.

## Author and journal declarations requiring human confirmation

- [ ] Confirm the author list and contribution statement against actual contributions, including code, analysis, clinical input, and manuscript drafting.
- [ ] Confirm affiliation, corresponding-author email, ORCID if used, and that each author has approved the final manuscript and venue.
- [ ] Confirm funding and acknowledgments; state none only if factually correct.
- [ ] Confirm competing-interest and AI-tool disclosure requirements for the selected journal and state only verified facts.
- [ ] Ask the relevant institution/ethics authority whether this retrospective secondary analysis of public data requires an ethics/IRB statement; do not invent an exemption or approval.
- [ ] Confirm all references, dataset versions, table/figure numbers, and claims against source artifacts.
- [ ] Choose the exact IEEE journal and apply its current submission requirements; the current IEEEtran source is a journal-style draft, not venue-final.
- [ ] Review whether the evidence-audit contribution is within the selected journal's scope and whether its novelty is sufficient.

## Current decision

**arXiv:** coherent source and score-level replay auditor exist, but a clean-clone reproduction bundle and the author/data-rights declarations are not complete. Do not call the release package complete yet.

**IEEE journal:** not submission-ready. The manuscript is a retrospective evidence audit, not a clinically validated image-to-diagnosis model. It reports exposed replay and internal development estimates, no independent external validation, and needs author-level novelty/scope review plus the reproducibility and declaration items above.

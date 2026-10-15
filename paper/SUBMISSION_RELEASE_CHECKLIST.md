# Submission and release checklist

Status is based on the repository and primary dataset-source pages checked on 2026-10-15. This is an author review checklist, not legal advice or institutional approval.

## Evidence and reproducibility

- [x] Re-execute the frozen policy on all 981 locally available score rows and compare every action and risk score.
- [x] Verify the replay source counts, 270/385 specificity numerator/denominator, zero replay clearances of referable or ungradable cases, and principal grouped/nested/transfer summary counts.
- [x] Emit input SHA-256 digests and a machine-readable audit report.
- [x] Provide a clean-clone aggregate-integrity check over tracked summaries; this is explicitly not row-level replay.
- [ ] Reproduction of the 981 row-level decisions from a clean public clone: blocked because the score rows and policy are local/untracked pending the rights review below.
- [ ] Reproduce component scores from raw images: blocked; not all checkpoints, preprocessing, and inference paths are packaged.
- [ ] Re-run every figure/table from raw predictions: incomplete; current auditor verifies headline values and exact replay, not every manuscript cell or plot.
- [x] Prepare a self-contained arXiv LaTeX upload archive without per-eye score/label files: `paper/submission/active_retina_arxiv_source_2026-10-15.zip`.
- [x] Compile both manuscript sources successfully with the desktop LaTeX compiler. This compiler does not export a local PDF artifact; arXiv accepts TeX source and generates a preview during its upload workflow.
- [ ] Before arXiv posting, inspect arXiv's own compiled preview and metadata, verify the archive is the intended final revision, and complete author-side submission steps.
- [ ] Create a final immutable release record after author and data-use checks. Keep the full score-level reproduction package separate unless redistribution rights are confirmed.
- [ ] Author must choose whether the project code remains `All rights reserved` or receives an explicit license. Do not describe the repository as open source while the current license is in force.

## Dataset rights and data availability

| Source | What the primary source says | Release decision |
|---|---|---|
| DDR/OIA-DDR | The official dataset repository states CC BY-NC-SA 4.0 for images, annotations, labels, and dataset materials; its README also says commercial use needs written permission. [Official repository](https://github.com/nkicsl/DDR-dataset) | Cite and link to original source; do not place data or derived per-eye score/label rows in this public repo until the author confirms the license implications and noncommercial scope. |
| DeepDRiD | The official repository provides a CC BY-SA 4.0 `LICENSE` and describes research use. [Official repository](https://github.com/deepdrdoc/DeepDRiD) | Confirm that the repository license applies to the dataset files used and determine whether derived row-level outputs/policy references may be redistributed under compatible terms. |
| DRTiD | The official repository says access is through an application form, describes the dataset as research-only, and requires citation; it does not display a separate machine-readable dataset license. [Official repository](https://github.com/FDU-VTS/DRTiD) | Confirm the grant/access terms permit publication and public redistribution of derived eye-level scores, labels, and policy artifacts. Until then, keep these files out of GitHub. |
| MMRDR | The data descriptor points to a Figshare record. [Data descriptor](https://www.nature.com/articles/s41597-026-07005-9) | The precise item-level license and derivative-output terms were not verified from the primary Figshare record in this review. Verify record metadata and any restrictions before release. |
| IDRiD | The manuscript records use in a later training experiment, but the applicable image/data license and redistribution permission are not established in this package. | Confirm the exact release terms and whether training-derived artifacts are distributable. |

These entries record source statements, not a legal determination. The author must confirm that the applicable access terms permit publication of the paper's aggregate results. Public availability of images does not by itself resolve whether row-level predictions, labels, identifiers, or dataset-derived rank references may be republished. The manuscript-only arXiv archive contains none of those row-level files; the separate full-replay release remains gated on rights review. No raw dataset images are included here.

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

**arXiv:** a self-contained source archive exists and both manuscript sources compile. The row-level replay bundle is not needed in the upload archive and remains a separate rights-gated reproducibility item. Before posting, the author must verify that data-use terms permit the reported aggregate results, review the exact source and arXiv-generated preview/metadata, and personally accept arXiv's submission/license terms. Thus the upload files are prepared, but author authorization and the arXiv-side preview/metadata check remain.

**IEEE journal:** the source is a journal-style evidence-audit draft, not a validated clinical-model paper. It reports exposed replay and internal development estimates, not independent external validation. A specific journal, its current template/requirements, author declarations, publication-rights confirmation, and author-level novelty/scope review remain before describing it as ready to submit.

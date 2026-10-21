import re
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
MANUSCRIPT = ROOT / "paper/active_retina_ieee_so_far.tex"
ARXIV_MANUSCRIPT = ROOT / "paper/active_retina_arxiv_full.tex"


class IeeeNoveltyPositioningTests(unittest.TestCase):
    def test_introduction_states_retrospective_questions_only_once(self):
        source = MANUSCRIPT.read_text(encoding="utf-8")
        intro = source.split(r"\section{Related Work}", maxsplit=1)[0]
        self.assertEqual(intro.count("Our retrospective questions are:"), 1)
        self.assertNotIn("We organize the audit around three retrospective questions.", intro)

    def test_source_ethics_are_attributed_and_not_conflated_with_secondary_analysis(self):
        source = MANUSCRIPT.read_text(encoding="utf-8")
        self.assertIn("QingYanLunShen[2024] No. 41", source)
        self.assertIn("waiver of informed consent for its retrospective collection", source)
        self.assertIn("participant informed consent was obtained", source)
        self.assertIn("do not establish the ethics determination, consent basis, or scope applicable to the present secondary analysis", source)
        self.assertIn("does not infer source-study procedures for DDR or DRTiD", source)

    def test_mmrdr_cfp_lineage_and_patient_overlap_are_not_overstated(self):
        source = MANUSCRIPT.read_text(encoding="utf-8")
        self.assertIn("identifies OIA-DDR as the source of its CFP subset", source)
        self.assertIn("matching suffixes alone do not establish same-eye correspondence", source)
        self.assertIn("exact patient overlap between datasets cannot be established", source)
        self.assertNotIn("CFP photographs from a distinct cohort", source)

    def test_mmrdr_item_license_is_not_presented_as_upstream_rights_clearance(self):
        source = MANUSCRIPT.read_text(encoding="utf-8")
        self.assertIn("the MMRDR item's displayed license is not treated here as a determination", source)
        self.assertIn(r"\cite{mmrdr,ddr}", source)
        self.assertIn("not as a legal interpretation of every dataset component", source)

    def test_public_reproducibility_limits_are_not_promised_as_supplemental_artifacts(self):
        source = MANUSCRIPT.read_text(encoding="utf-8")
        self.assertIn("checksums for its aggregate audit artifacts and an MMRDR run card", source)
        self.assertIn("does not include a complete record-level source-partition map", source)
        self.assertIn("cannot reconstruct the row-level cohorts, execute the score replay, or reproduce image inference", source)
        self.assertIn("does not claim that its supplement completes the provenance trail", source)
        self.assertNotIn("A submission supplement should include checksums", source)

    def test_latex_emphasis_command_is_not_rendered_as_literal_text(self):
        source = MANUSCRIPT.read_text(encoding="utf-8")
        self.assertIn(r"\emph{true} specificity", source)
        self.assertIsNone(re.search(r"(?<!\\)emph\{true\} specificity", source))

    def test_diagnostic_multiple_comparison_prior_art_is_cited_and_bounded(self):
        source = MANUSCRIPT.read_text(encoding="utf-8")
        self.assertIn(r"\cite{westphal2024}", source)
        self.assertEqual(source.count(r"\bibitem{westphal2024}"), 1)
        self.assertIn("doi: 10.1177/09622802241236933", source)
        self.assertIn("We do not implement, extend, or empirically compare those procedures.", source)
        self.assertIn("not a general correction for selection bias.", source)
        self.assertIn(r"\begin{thebibliography}{30}", source)

    def test_dr_operating_point_transfer_prior_art_is_cited_and_distinguished(self):
        source = MANUSCRIPT.read_text(encoding="utf-8")
        self.assertIn(r"\cite{bhuiyan2026}", source)
        self.assertEqual(source.count(r"\bibitem{bhuiyan2026}"), 1)
        self.assertIn("10.1109/PECCII70991.2026.11661985", source)
        self.assertIn("69.41\\% sensitivity at high specificity", source)
        self.assertIn("93.42\\% sensitivity and 84.30\\% specificity", source)
        self.assertIn("rules out any claim that the present study is the first DR analysis of threshold transfer", source)
        self.assertIn("These designs answer different questions and are not comparative model evaluations", source)

    def test_2026_uwf_and_mbrset_prior_art_is_cited_with_task_boundary(self):
        for manuscript in (MANUSCRIPT, ARXIV_MANUSCRIPT):
            with self.subTest(manuscript=manuscript.name):
                source = manuscript.read_text(encoding="utf-8")
                self.assertIn(r"\cite{li2026uwf}", source)
                self.assertIn(r"\cite{li2026mbrset}", source)
                self.assertEqual(source.count(r"\bibitem{li2026uwf}"), 1)
                self.assertEqual(source.count(r"\bibitem{li2026mbrset}"), 1)
                self.assertIn("10.1016/j.ijmedinf.2026.106503", source)
                self.assertIn("10.64898/2026.04.17.26351092", source)
                self.assertIn("different cohorts, endpoints, models, and estimands", source)
                self.assertIn("not performance comparators", source)

    def test_frozen_encoder_calibration_prior_art_bounds_novelty(self):
        for manuscript in (MANUSCRIPT, ARXIV_MANUSCRIPT):
            with self.subTest(manuscript=manuscript.name):
                source = manuscript.read_text(encoding="utf-8")
                self.assertIn(r"\cite{poyrazer2026frozen}", source)
                self.assertEqual(source.count(r"\bibitem{poyrazer2026frozen}"), 1)
                self.assertIn("10.3389/fmed.2026.1815982", source)
                self.assertIn("frozen-representation transfer and calibration", source)
                self.assertIn("make no claim to introduce either question", source)
                self.assertIn("not performance comparators", source)
                self.assertIn("study-trained six-logit head", source)
                self.assertNotIn("trained ordinal head on MMRDR-UWF", source)

    def test_arxiv_reports_simulation_as_synthetic_mechanism_not_retinal_evidence(self):
        source = ARXIV_MANUSCRIPT.read_text(encoding="utf-8")
        self.assertIn("Controlled zero-miss candidate-search simulation", source)
        self.assertIn("identical-channel negative control", source)
        self.assertIn("2,000 replicates per condition", source)
        self.assertIn("55.45\\%", source)
        self.assertIn("0/2,000", source)
        self.assertIn("not clinical evidence or an explanation of the observed retinal results", source)
        self.assertIn(r"\label{fig:arxiv-panel-simulation}", source)
        self.assertIn(r"\label{tab:arxiv-panel-simulation}", source)

    def test_recent_dr_decision_and_external_transfer_prior_art_is_bounded(self):
        for manuscript in (MANUSCRIPT, ARXIV_MANUSCRIPT):
            with self.subTest(manuscript=manuscript.name):
                source = manuscript.read_text(encoding="utf-8")
                self.assertIn(r"\cite{jena2026}", source)
                self.assertIn(r"\cite{sheng2026orderdr}", source)
                self.assertEqual(source.count(r"\bibitem{jena2026}"), 1)
                self.assertEqual(source.count(r"\bibitem{sheng2026orderdr}"), 1)
                self.assertIn("doi: 10.7759/cureus.108630", source)
                self.assertIn("doi: 10.3389/fendo.2026.1923216", source)
        ieee = MANUSCRIPT.read_text(encoding="utf-8")
        self.assertIn("17.5\\% specificity", ieee)
        self.assertIn("95\\% sensitivity", ieee)
        self.assertIn("nearly 25\\% relative to the single-model 95\\% setting", ieee)
        self.assertIn("cannot be ranked against one another", ieee)
        self.assertIn("Our frozen MMRDR threshold is a within-dataset held-out evaluation, not external transfer", ieee)
        arxiv = ARXIV_MANUSCRIPT.read_text(encoding="utf-8")
        self.assertIn("17.5\\% specificity", arxiv)
        self.assertIn("95\\% sensitivity", arxiv)
        self.assertIn("nearly 25\\% relative to the single-model 95\\% setting", arxiv)
        self.assertIn("must not be interpreted as a performance comparison", arxiv)
        self.assertIn("not external transfer", arxiv)

    def test_approved_systems_meta_analysis_is_context_not_a_comparator(self):
        for manuscript in (MANUSCRIPT, ARXIV_MANUSCRIPT):
            with self.subTest(manuscript=manuscript.name):
                source = manuscript.read_text(encoding="utf-8")
                self.assertIn(r"\cite{wang2026approveddr}", source)
                self.assertEqual(source.count(r"\bibitem{wang2026approveddr}"), 1)
                self.assertIn("82 studies, 887,244 examinations, 25 devices, and 28 countries", source)
                self.assertIn("0.93/0.90 per patient and 0.92/0.93 per eye", source)
                self.assertIn("disease threshold, national-income level, image gradability", source)
                comparator_boundary = (
                    "not a comparator for this study"
                    if manuscript == MANUSCRIPT
                    else "not a performance comparator for ActiveRetina"
                )
                self.assertIn(comparator_boundary, source)
                self.assertIn("10.1038/s41746-025-02223-8", source)

    def test_radiomics_leakage_prior_art_is_quantified_and_novelty_bounded(self):
        source = MANUSCRIPT.read_text(encoding="utf-8")
        self.assertIn(r"\cite{garciahidalgo2026}", source)
        self.assertIn("50 cases and 1,000 features", source)
        self.assertIn("fully leaked pipeline reached AUC 1.000 while the properly nested reference was 0.493", source)
        self.assertIn("do not claim to introduce medical-imaging leakage audits", source)
        self.assertIn("not a new general method", source)

    def test_cross_modality_operating_point_collapse_prior_art_is_cited_and_bounded(self):
        source = MANUSCRIPT.read_text(encoding="utf-8")
        self.assertIn(r"\cite{abdikenov2026}", source)
        self.assertEqual(source.count(r"\bibitem{abdikenov2026}"), 1)
        self.assertIn("doi: 10.3390/info17100954", source)
        self.assertIn("not evidence for the same mechanism in retinal imaging", source)
        self.assertIn("We do not claim that operating-point collapse", source)

    def test_core_selective_prediction_and_selection_references_have_primary_links(self):
        source = MANUSCRIPT.read_text(encoding="utf-8")
        expected = {
            "geifman2017": "https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html",
            "selectivenet": "https://proceedings.mlr.press/v97/geifman19a.html",
            "cawley2010": "https://www.jmlr.org/papers/v11/cawley10a.html",
            "galappaththige2024": "https://openaccess.thecvf.com/content/WACV2024/html/Galappaththige_Generalizing_to_Unseen_Domains_in_Diabetic_Retinopathy_Classification_WACV_2024_paper.html",
        }
        for key, url in expected.items():
            with self.subTest(reference=key):
                self.assertEqual(source.count(rf"\bibitem{{{key}}}"), 1)
                self.assertIn(url, source)
        self.assertIn("no. 70, pp. 2079--2107", source)
        self.assertIn("PMLR, vol. 97, 2019, pp. 2151--2159", source)

    def test_date_shaped_values_are_identified_as_rng_seeds(self):
        manuscripts = [path for path in (MANUSCRIPT, ARXIV_MANUSCRIPT) if path.exists()]
        self.assertIn(MANUSCRIPT, manuscripts)
        for manuscript in manuscripts:
            source = manuscript.read_text(encoding="utf-8")
            self.assertIn("fixed integer RNG seeds 20261015--20261019", source)
            self.assertIn("not calendar dates", source)
            self.assertIn("first value identifies the original split", source)
            self.assertIn("four additional", source)
            self.assertIn("RNG seed ID (last two digits)", source)


if __name__ == "__main__":
    unittest.main()

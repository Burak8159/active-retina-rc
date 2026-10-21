import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]


def _table_body(source: str, label: str) -> str:
    start = source.index(f"\\label{{{label}}}")
    end = source.index("\\end{table", start)
    return source[start:end]


def test_official_test_table_matches_run_card_and_keeps_claim_boundary():
    run_card = json.loads(
        (ROOT / "paper/reproducibility/mmrdr_uwf_official_test_run_card.json").read_text()
    )
    result = run_card["official_test_result"]
    manuscript = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    table = _table_body(manuscript, "tab:replay-model-contrast")

    expected_fragments = (
        f"{result['true_positive'] + result['false_negative']:,}",
        f"{100 * result['specificity']:.2f}\\% specificity ({result['true_negative']:,}/{result['true_negative'] + result['false_positive']:,};",
        f"{result['false_negative']}/{result['true_positive'] + result['false_negative']:,}",
    )
    assert all(fragment in manuscript for fragment in expected_fragments)
    assert result["clinical_certification"] is False
    assert "primary fixed-threshold, image-level within-dataset outcome" in table.lower()


def test_ieee_title_states_a_retrospective_evidence_audit():
    source = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    title = next(line for line in source.splitlines() if line.startswith(r"\title{"))

    assert "Selection History" in title
    assert "Operating-Point Evaluation" in title
    assert "Retrospective Evidence Audit" in title


def test_abstracts_center_the_locked_result_and_bound_posthoc_auc():
    for name in ("active_retina_ieee_so_far.tex", "active_retina_arxiv_full.tex"):
        source = (ROOT / "paper" / name).read_text()
        abstract = source.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
        assert "99.61\\%" in abstract
        assert "20.02\\%" in abstract
        assert "five false-negative images" in abstract
        assert "Post-hoc" in abstract and "deployable threshold" in abstract
        assert "deployable threshold" in abstract
        assert (
            "clinical-use claim" in abstract
            or "clinical-utility claim" in abstract
            or "clinical-validation claim" in abstract
        )
        assert "High Image-Level Discrimination" not in source


def test_abstracts_distinguish_zero_miss_failure_from_specificity_collapse():
    for name in ("active_retina_ieee_so_far.tex", "active_retina_arxiv_full.tex"):
        source = (ROOT / "paper" / name).read_text()
        abstract = source.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
        assert "20.50\\%" in abstract
        assert "20.02\\%" in abstract
        assert "zero-miss condition" in abstract
        assert "-0.48 percentage points" in abstract


def test_ieee_abstract_keeps_raw_image_and_score_replay_evidence_separate():
    source = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    abstract = source.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
    assert "two non-comparable evidence objects" in abstract
    assert "development-selected threshold" in abstract
    assert "separate, exposed precomputed-score policy replay" in abstract
    assert "Separately, deterministic replay" in abstract
    assert "Among 596 referable-or-ungradable eye records, three decisions changed from REFER to CLEAR" in abstract
    assert "paired eye-level missed-required-eye-rate difference" in abstract


def test_paired_replay_crossfit_receipt_matches_both_manuscripts_and_transition_tables():
    receipt = json.loads(
        (ROOT / "paper/reproducibility/paired_replay_crossfit_audit_20261020.json").read_text()
    )
    paired = receipt["paired_policy_comparison"]
    assert receipt["status"] == "CONDITIONAL_EXPOSED_COHORT_PAIRED_AUDIT_NOT_VALIDATION"
    assert paired["replay_true_negatives"] == 270
    assert paired["crossfit_true_negatives"] == 268
    assert paired["eligible_negative_eyes"] == 385
    assert receipt["required_case_misses"] == {
        "deterministic_replay": 0,
        "patient_grouped_crossfit": 3,
    }
    miss = paired["required_case_miss_comparison"]
    assert miss["required_case_eyes"] == 596
    assert miss["required_case_patient_clusters"] == 574
    assert miss["replay_misses"] == 0
    assert miss["crossfit_misses"] == 3
    assert miss["difference_percentage_points"] == pytest.approx(0.5033557047)
    assert miss["patient_cluster_bootstrap"]["difference_95_percentile_interval_pp"] == pytest.approx([0.0, 1.1744966443])
    assert receipt["all_eye_action_transitions"] == {
        "replay_clear__crossfit_clear": 268,
        "replay_clear__crossfit_refer": 2,
        "replay_refer__crossfit_clear": 3,
        "replay_refer__crossfit_refer": 708,
    }
    assert receipt["case_identifiers_emitted"] is False
    for name in ("active_retina_ieee_so_far.tex", "active_retina_arxiv_full.tex"):
        source = (ROOT / "paper" / name).read_text()
        table = _table_body(source, "tab:paired-transitions")
        methods = source.split(r"\section{Results}", 1)[0]
        for value in ("268 & 2 & 0 & 115", "0 & 0 & 3 & 593"):
            assert value in table
        assert "For the required-eye miss-rate difference, we separately resampled the 574 source--patient clusters" in methods
        assert "The estimand remains eye-level; patient-cluster resampling changes the resampling unit, not the endpoint." in methods
        assert "$-0.52$" in source
        assert "$[-1.31,0.00]$" in source
        assert "$+0.50$" in source
        assert "$[0.00,1.17]$" in source
        assert "missed-eye interval includes zero" in source
    reproduction = (ROOT / "paper/reproducibility/README.md").read_text()
    assert "autonomy_route_three_dataset_manifest_20261011.csv" in reproduction
    assert "autonomy_route_three_dataset_manifest_20261011_exactdedup.csv" not in reproduction


def test_ieee_abstract_and_keywords_fit_general_ieee_guidance():
    source = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    abstract = source.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
    keywords = source.split(r"\begin{keywords}", 1)[1].split(r"\end{keywords}", 1)[0]

    assert len(abstract.split()) <= 250
    assert len([item for item in keywords.split(",") if item.strip()]) in range(3, 6)


def test_replay_model_contrast_table_remains_full_width():
    source = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    label = r"\label{tab:replay-model-contrast}"
    label_position = source.index(label)
    table_start = source.rfind(r"\begin{table*}", 0, label_position)
    single_column_start = source.rfind(r"\begin{table}", 0, label_position)
    table_end = source.index(r"\end{table*}", label_position)

    assert table_start > single_column_start
    assert "{\\textwidth}" in source[table_start:table_end]


def test_submission_index_identifies_the_current_ieee_review_snapshot_consistently():
    readme = (ROOT / "paper/submission/README.md").read_text()
    bundle = "active_retina_ieee_review_bundle_2026-10-19_v116.zip"

    assert bundle in readme
    assert "v87 archive above is the current review snapshot" not in readme
    assert "verification_2026-10-19_v116.json" in readme
    assert "active_retina_arxiv_source_2026-10-19_v30.zip" in readme
    assert "A paired analysis of the same exposed cohort" in readme
    assert "mmrdr_test_score_diagnostic_20261019.json" in readme
    assert "Only v62 is identified here as the current review snapshot" not in readme


def test_bibliographies_follow_first_citation_order_in_both_manuscripts():
    import re

    for manuscript in (
        ROOT / "paper/active_retina_ieee_so_far.tex",
        ROOT / "paper/active_retina_arxiv_full.tex",
    ):
        source = manuscript.read_text()
        body, bibliography = source.split(r"\begin{thebibliography}", 1)
        citation_order = []
        for group in re.findall(r"\\cite[A-Za-z]*\{([^}]*)\}", body):
            for key in group.split(","):
                key = key.strip()
                if key and key not in citation_order:
                    citation_order.append(key)
        reference_order = re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]*)\}", bibliography)
        assert reference_order == citation_order, manuscript.name


def test_ieee_ai_disclosure_names_assisted_sections_and_assistance_level():
    source = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    start = source.index(r"\section*{Acknowledgment and Generative-AI Disclosure}")
    end = source.index(r"\begin{thebibliography}", start)
    disclosure = source[start:end]

    assert "OpenAI ChatGPT, accessed through Codex" in disclosure
    for section in (
        "Abstract",
        "Introduction",
        "Related Work",
        "Methods",
        "Results",
        "Discussion",
        "Limitations",
        "Conclusion",
        "Reproducibility and Data Availability",
    ):
        assert section in disclosure
    assert "outlining, drafting, restructuring, and revising" in disclosure
    assert "implementing and testing analysis and audit code" in disclosure
    assert "sole human author is responsible for verifying" in disclosure


def test_ieee_access_abstract_contains_no_numbered_reference_citation():
    source = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    abstract = source.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]

    assert r"\cite{" not in abstract


def test_ieee_access_ai_assistance_is_cited_in_assisted_body_sections():
    source = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    body = source.split(r"\begin{abstract}", 1)[1].split(r"\section{Author and Contribution Statement}", 1)[0]
    for section in ("Introduction", "Related Work", "Methods", "Results", "Discussion", "Limitations", "Conclusion"):
        start = body.index(rf"\section{{{section}}}")
        following = body.find(r"\section", start + 1)
        section_text = body[start:] if following < 0 else body[start:following]
        assert r"\cite{openaiChatGPT}" in section_text, section


def test_source_mixture_standardization_matches_cluster_bootstrap_receipt():
    receipt = json.loads(
        (ROOT / "paper/reproducibility/exposed_replay_patient_cluster_bootstrap_20261019.json").read_text()
    )
    mixture = receipt["source_mix_sensitivity"]
    source = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    table = _table_body(source, "tab:source-mixture")

    assert mixture["observed_mix_specificity"] == pytest.approx(270 / 385)
    assert mixture["equal_source_standardized_specificity"] == pytest.approx(
        (134 / 206 + 51 / 52 + 85 / 127) / 3
    )
    assert "70.13\\%" in table
    assert "76.68\\%" in table
    assert "72.93--80.32" in table
    assert "58.25--71.36" in table
    assert "93.75--100.00" in table
    contrasts = mixture["exploratory_pairwise_source_contrasts"]
    deep_minus_ddr = contrasts["ddr_hf_specificity300_minus_deepdrid_paired"]
    deep_minus_drtid = contrasts["deepdrid_paired_minus_drtid_paired"]
    assert deep_minus_ddr["point_difference"] == pytest.approx(-0.3302837938760268)
    assert deep_minus_drtid["point_difference"] == pytest.approx(0.31147789218655353)
    assert "33.03 percentage points" in source
    assert "31.15 points" in source
    contrast_table = _table_body(source, "tab:source-contrasts")
    assert "33.03" in contrast_table and "25.24--40.29" in contrast_table
    assert "31.15" in contrast_table and "22.05--40.16" in contrast_table
    assert "-1.88" in contrast_table and "-12.31" in contrast_table
    assert "not a target-clinic estimate" in mixture["interpretation"]

    ieee_figure = source.split(r"\label{fig:source-mixture}", 1)[0].rsplit(r"\begin{figure}", 1)[-1]
    arxiv = (ROOT / "paper/active_retina_arxiv_full.tex").read_text()
    arxiv_bootstrap = _table_body(arxiv, "tab:source-bootstrap")
    for value in ("58.25--71.36", "93.75--100.00", "58.27--74.80", "25.24--40.29", "22.05--40.16"):
        assert value in arxiv_bootstrap
    assert "\\widehat{\\mathrm{Sp}}_{\\mathrm{equal}}" in arxiv
    assert "100\\sum_d(1/3-\\widehat{w}_d)" in arxiv
    assert "$-13.12$-point" in arxiv and "$+19.45$-point" in arxiv
    assert "$+0.23$-point" in arxiv
    assert "$-13.12$ points" in source and "$+19.45$ points" in source
    arxiv_figure = arxiv.split(r"\label{fig:source}", 1)[0].rsplit(r"\begin{figure}", 1)[-1]
    for value in ("65.05", "98.08", "66.93", "70.13", "76.68"):
        assert value in ieee_figure
        assert value in arxiv_figure
    assert "72.93--80.32\\%" in arxiv
    assert "not a model improvement" in arxiv


def test_manuscript_discloses_source_specific_dataset_rights_without_claiming_clearance():
    source = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    start = source.index("\\subsection{Data Provenance and Source Harmonization}")
    next_subsection = source.index("\\subsection{", start + len("\\subsection{"))
    paragraph = source[start:next_subsection]
    for notice in (
        "CC BY 4.0",
        "CC BY-NC-SA 4.0",
        "CC BY-SA 4.0",
        "research-only, application-based access",
        "not as a legal interpretation",
        "no raw images or per-eye score/label rows",
    ):
        assert notice in paragraph
    for repo_url in (
        "https://figshare.com/articles/dataset/MMRDR/29423747",
        "https://github.com/nkicsl/DDR-dataset",
        "https://github.com/deepdrdoc/DeepDRiD",
        "https://github.com/FDU-VTS/DRTiD",
    ):
        assert repo_url in paragraph


def test_ieee_access_front_matter_uses_template_placeholders_not_pseudo_publication_metadata():
    source = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    assert r"\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}" in source
    assert r"\doi{10.1109/ACCESS.2017.DOI}" in source
    assert r"\vol{XX}" in source
    assert r"\year{2026}" in source
    assert "Not assigned (preprint)" not in source
    assert "Preprint manuscript, October 2026" not in source


def test_author_biography_does_not_assert_unconfirmed_research_interests():
    source = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    start = source.index(r"\begin{IEEEbiographynophoto}{Burak Ata Aksu}")
    end = source.index(r"\end{IEEEbiographynophoto}", start)
    biography = source[start:end]

    assert "sole human author" in biography
    assert "require author verification before submission" in biography
    assert "research interests include" not in biography


def test_official_test_confusion_row_matches_run_card():
    run_card = json.loads(
        (ROOT / "paper/reproducibility/mmrdr_uwf_official_test_run_card.json").read_text()
    )
    result = run_card["official_test_result"]
    table = _table_body(
        (ROOT / "paper/active_retina_ieee_so_far.tex").read_text(),
        "tab:official-test",
    )

    expected = (
        f"Official UWF test (all rows) & {run_card['dataset']['official_test_rows']}"
        f" & {result['true_positive'] + result['false_negative']}"
        f" & {result['true_positive']} & {result['false_negative']}"
        f" & {result['true_negative'] + result['false_positive']}"
        f" & {result['true_negative']} & {result['false_positive']}"
    )
    assert expected in table


def test_mmrdr_grade5_logit_limitation_is_explicit_and_bounded():
    run_card = json.loads(
        (ROOT / "paper/reproducibility/mmrdr_uwf_official_test_run_card.json").read_text()
    )
    model = run_card["reconstructed_training_provenance"]["loss_semantics"]
    assert model["grade_index_5_has_mmrdr_training_examples"] is False
    assert model["referable_logit_includes_index_5"] is True

    for relative in (
        "paper/active_retina_ieee_so_far.tex",
        "paper/active_retina_arxiv_full.tex",
    ):
        source = (ROOT / relative).read_text()
        assert "no direct grade-5 categorical supervision" in source
        assert "direction and magnitude are unknown" in source
        assert "pairwise-ranking, and hard-boundary losses" in source


def test_arxiv_grade_specific_false_referral_rates_match_the_aggregate_diagnostic():
    diagnostic = json.loads(
        (ROOT / "paper/reproducibility/mmrdr_test_score_diagnostic_20261019.json").read_text()
    )
    rates = {row["grade"]: row for row in diagnostic["grade_stratified"]}
    assert rates[0]["n"] == 800
    assert rates[0]["refer_rate_at_frozen_threshold"] == pytest.approx(593 / 800)
    assert rates[1]["n"] == 509
    assert rates[1]["refer_rate_at_frozen_threshold"] == pytest.approx(454 / 509)

    arxiv = (ROOT / "paper/active_retina_arxiv_full.tex").read_text()
    assert "593/800 grade-0 images (74.13\\% referred)" in arxiv
    assert "454/509 grade-1 images (89.19\\% referred)" in arxiv


def test_official_test_derived_metrics_match_confusion_counts_in_both_manuscripts():
    run_card = json.loads(
        (ROOT / "paper/reproducibility/mmrdr_uwf_official_test_run_card.json").read_text()
    )
    result = run_card["official_test_result"]
    tp, fn = result["true_positive"], result["false_negative"]
    tn, fp = result["true_negative"], result["false_positive"]
    positives, negatives = tp + fn, tn + fp
    total = positives + negatives
    sensitivity = tp / positives
    specificity = tn / negatives
    accuracy = (tp + tn) / total
    balanced_accuracy = (sensitivity + specificity) / 2
    ppv = tp / (tp + fp)
    referral_count = tp + fp

    z = 1.959963984540054
    denominator = 1 + z**2 / negatives
    center = (specificity + z**2 / (2 * negatives)) / denominator
    radius = z * (
        specificity * (1 - specificity) / negatives
        + z**2 / (4 * negatives**2)
    ) ** 0.5 / denominator
    ci_low, ci_high = center - radius, center + radius
    sens_denominator = 1 + z**2 / positives
    sens_center = (sensitivity + z**2 / (2 * positives)) / sens_denominator
    sens_radius = z * (
        sensitivity * (1 - sensitivity) / positives
        + z**2 / (4 * positives**2)
    ) ** 0.5 / sens_denominator
    sens_ci_low, sens_ci_high = sens_center - sens_radius, sens_center + sens_radius

    expected_metrics = (
        f"{100 * accuracy:.2f}\\%",
        f"{100 * balanced_accuracy:.2f}\\%",
        f"{100 * referral_count / total:.2f}\\%",
        f"{100 * sens_ci_low:.2f}--{100 * sens_ci_high:.2f}\\%",
        f"{100 * ci_low:.2f}--{100 * ci_high:.2f}\\%",
    )
    checked_manuscripts = 0
    for path in (
        "paper/active_retina_ieee_so_far.tex",
        "paper/active_retina_arxiv_full.tex",
    ):
        if not (ROOT / path).exists():
            continue
        source = (ROOT / path).read_text()
        checked_manuscripts += 1
        normalized = source.replace(",", "")
        for metric in expected_metrics:
            assert metric in normalized, f"{path} is missing derived result: {metric}"
        assert f"{referral_count}/{total}" in normalized
    assert checked_manuscripts >= 1
    assert f"positive predictive value {100 * ppv:.2f}\\%" in (
        ROOT / "paper/active_retina_ieee_so_far.tex"
    ).read_text()
    assert round(sensitivity, 4) == 0.9961
    assert round(specificity, 4) == 0.2002


def test_official_test_grade_table_matches_aggregate_run_card():
    run_card = json.loads(
        (ROOT / "paper/reproducibility/mmrdr_uwf_official_test_run_card.json").read_text()
    )
    grades = run_card["official_test_grade_actions"]
    table = _table_body(
        (ROOT / "paper/active_retina_ieee_so_far.tex").read_text(),
        "tab:grade-errors",
    )

    for row in grades:
        images = row["images"]
        action = "specificity" if row["grade"] < 2 else "sensitivity"
        rate_numerator = row["cleared"] if row["grade"] < 2 else row["referred"]
        expected = (
            f"{row['grade']} & {images} & {row['referred']} & {row['cleared']}"
            f" & {100 * rate_numerator / images:.2f}\\% {action}"
        )
        assert expected in table

    result = run_card["official_test_result"]
    assert sum(row["images"] for row in grades) == run_card["dataset"]["official_test_rows"]
    assert sum(row["referred"] for row in grades) == result["true_positive"] + result["false_positive"]
    assert sum(row["cleared"] for row in grades) == result["true_negative"] + result["false_negative"]


def test_mmrdr_development_table_matches_retained_candidate_summaries():
    source = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    table = _table_body(source, "tab:mmrdr")

    expected = []
    for path, row_label in (
        ("results/mmrdr_uwf_cached_head_20261013/summary.json", "ResNet-50 cached UWF head"),
        ("results/mmrdr_uwf_dino_head_dev_20261013/summary.json", "DINO frozen UWF head"),
        ("results/mmrdr_uwf_dino_hardhead_dev_20261013/summary.json", "DINO hard-head UWF candidate"),
    ):
        summary = json.loads((ROOT / path).read_text())
        metrics = summary["validation_metrics_at_zero_miss_threshold"]
        expected.append(
            f"{row_label} & {metrics['true_negative']} & "
            f"{100 * metrics['specificity']:.2f}\\%"
        )

    embeddings = json.loads(
        (ROOT / "results/mmrdr_embedding_heads_20261013/summary.json").read_text()
    )
    hgb = next(
        row for row in embeddings["results_ranked_by_zero_miss_specificity"]
        if row["model"] == "hist_gradient_boost"
    )
    expected.append(
        f"ResNet+DINO feature stack, HGB & {hgb['true_negative']} & "
        f"{100 * hgb['specificity_zero_miss']:.2f}\\%"
    )

    retfound = json.loads(
        (ROOT / "results/mmrdr_uwf_retfound_eyepacs_heads_20261013/summary.json").read_text()
    )
    retfound_models = {
        "hist_gradient_boost": "RETFound-EyePACS features, HGB",
        "rbf_svc_l2_c10": "RETFound-EyePACS features, RBF-SVM",
        "logreg_scaled_c0.01": "RETFound-EyePACS features, logistic head",
    }
    for row in retfound["results_ranked_by_zero_miss_specificity"]:
        if row["model"] in retfound_models:
            expected.append(
                f"{retfound_models[row['model']]} & {row['true_negative']} & "
                f"{100 * row['specificity_zero_miss']:.2f}\\%"
            )

    fusion = json.loads(
        (ROOT / "results/mmrdr_retfound_fusion_20261013/summary.json").read_text()
    )
    fusion_models = {
        "reference_hgb075_ft025": "Internal reference rank fusion",
        "reference_plus_logreg_scaled_c0.01_0.25": "Reference + 25\\% RETFound-logistic rank",
    }
    for row in fusion["candidates_ranked"]:
        if row["candidate"] in fusion_models:
            expected.append(
                f"{fusion_models[row['candidate']]} & {row['true_negative']} & "
                f"{100 * row['specificity_zero_miss']:.2f}\\%"
            )

    assert len(expected) == 9
    assert all(row in table for row in expected)
    assert "& 47 & 7.83\\%" not in table


def test_finetune_table_matches_saved_balanced_run_history():
    summary = json.loads(
        (ROOT / "results/combined_mmrdr_qualitygate_balanced_allft_20261013/summary.json").read_text()
    )
    table = _table_body(
        (ROOT / "paper/active_retina_ieee_so_far.tex").read_text(), "tab:ft"
    )
    rows = summary["history"]
    for item in rows:
        metrics = item["validation_zero_miss"]
        epoch = "Warm start (0)" if item["epoch"] == 0 else str(item["epoch"])
        expected = (
            f"{epoch} & {metrics['true_negative']} & "
            f"{100 * metrics['specificity']:.2f}\\% & "
            f"{metrics['false_negative_clearance'] + metrics['missed_ungradable']}"
        )
        assert expected in table
        assert metrics["negative"] == 386
        assert metrics["positive"] == 463
        assert metrics["ungradable"] == 133
        assert metrics["action_coverage"] == 1.0


def test_nested_fold_table_matches_saved_fold_records():
    summary = json.loads(
        (ROOT / "results/nested_crossfit_expert_weights_20261013/summary.json").read_text()
    )
    table = _table_body(
        (ROOT / "paper/active_retina_ieee_so_far.tex").read_text(), "tab:nestedfold"
    )
    folds = summary["fold_results"]
    for row in folds:
        weights = ", ".join(f"{weight:.1f}" for weight in row["weights"])
        expected = (
            f"{row['source']} & {row['fold']} & {row['heldout_patients']}"
            f" & {row['negatives']} & {row['true_negatives']} & {row['misses']}"
            f" & ({weights}) & {row['threshold']:.5f}"
        )
        assert expected in table
    assert sum(row["true_negatives"] for row in folds) == summary["true_negatives"]
    assert sum(row["misses"] for row in folds) == summary["misses_referable_or_ungradable"]


def test_pooled_and_source_grouped_tables_match_included_summaries():
    replay = json.loads(
        (ROOT / "results/deployable_exposed_baseline_7013_20261013_with_components/summary.json").read_text()
    )["same_cohort_replay_metrics"]
    crossfit = json.loads(
        (ROOT / "results/crossfit_expert_calibration_20261013_corrected/summary.json").read_text()
    )
    nested = json.loads(
        (ROOT / "results/nested_crossfit_expert_weights_20261013/summary.json").read_text()
    )
    nested_margin = json.loads(
        (ROOT / "results/nested_crossfit_expert_weights_margin_20261013/summary.json").read_text()
    )
    source = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    pooled = _table_body(source, "tab:cv")
    source_grouped = _table_body(source, "tab:sourcegrouped")
    fixed_folds = _table_body(source, "tab:fixedfold")

    replay_sp = 100 * replay["true_negatives"] / replay["n_gradable_nonreferable"]
    assert f"Same-cohort replay & {replay['true_negatives']} & {replay_sp:.2f}\\% & 0 & exposed" in pooled
    assert (
        f"Fixed blend, 5-fold & {crossfit['crossfit']['true_negatives']}"
        f" & {crossfit['crossfit']['specificity_percent']:.2f}\\%"
        f" & {crossfit['crossfit']['misses_referable_or_ungradable']}"
    ) in pooled
    assert (
        f"Nested weights, 5$\\times$4 & {nested['true_negatives']}"
        f" & {nested['specificity_percent']:.2f}\\%"
        f" & {nested['misses_referable_or_ungradable']}"
    ) in pooled
    assert (
        f"Nested + margin & {nested_margin['true_negatives']}"
        f" & {nested_margin['specificity_percent']:.2f}\\%"
        f" & {nested_margin['misses_referable_or_ungradable']}"
    ) in pooled

    replay_by_source = replay["per_source"]
    grouped_by_source = {}
    for fold in crossfit["fold_results"]:
        item = grouped_by_source.setdefault(
            fold["source"], {"tn": 0, "misses": 0}
        )
        item["tn"] += fold["true_negatives"]
        item["misses"] += fold["misses_referable_or_ungradable"]
    for name in ("DDR", "DeepDRiD", "DRTiD"):
        baseline = replay_by_source[name]
        grouped = grouped_by_source[name]
        expected = (
            f"{name} & {baseline['gradable_nonreferable']}"
            f" & {baseline['true_negatives']} & 0"
            f" & {grouped['tn']} & {grouped['misses']}"
        )
        assert expected in source_grouped

    for fold in crossfit["fold_results"]:
        expected = (
            f"{fold['source']} & {fold['fold']} & {fold['heldout_cases']}"
            f" & {fold['heldout_negatives']} & {fold['true_negatives']}"
            f" & {fold['misses_referable_or_ungradable']}"
        )
        assert expected in fixed_folds


def test_candidate_panel_table_matches_seed_level_aggregate():
    with (ROOT / "paper/submission/candidate_panel_seed_summary_2026-10-16.csv").open() as f:
        rows = list(csv.DictReader(f))
    table = _table_body(
        (ROOT / "paper/active_retina_ieee_so_far.tex").read_text(),
        "tab:candidatesensitivity",
    )

    for panel, label in (
        ("label_screened_oracle_contributors", "Oracle-contributor shortlist"),
        ("all_complete_aligned", "All complete aligned columns"),
    ):
        selected = [row for row in rows if row["candidate_panel"] == panel]
        mean_tn = sum(int(row["true_negatives"]) for row in selected) / len(selected)
        mean_misses = sum(int(row["referable_misses"]) for row in selected) / len(selected)
        mean_specificity = sum(float(row["specificity_percent"]) for row in selected) / len(selected)
        zero_miss_runs = sum(int(row["referable_misses"]) == 0 for row in selected)
        assert f"{label} &" in table
        assert f"& {mean_tn:.1f} & {mean_specificity:.2f}\\% & {mean_misses:.1f}; {zero_miss_runs}/5 &" in table
        assert len(selected) == 5


def test_crossfit_summary_is_described_as_post_selection_not_validation():
    summary = json.loads(
        (ROOT / "paper/submission/mmrdr_resnet50_threshold_crossfit_aggregate_2026-10-16.json").read_text()
    )
    assert summary["pooled_out_of_fold"] == {"tp": 660, "tn": 466, "fp": 99, "fn": 113}
    assert summary["coverage"] == 1.0
    assert summary["clinical_validation"] is False
    assert "post_selection" in summary["status"].lower()

    manuscript_paths = (
        "paper/active_retina_ieee_so_far.tex",
        "paper/active_retina_arxiv_full.tex",
    )
    for path in manuscript_paths:
        if not (ROOT / path).exists():
            continue
        source = (ROOT / path).read_text().lower()
        assert "84.16\\%" in source
        assert "113" in source
        assert "post-selection" in source
        assert "not independent" in source or "not an unbiased" in source


def test_local_full_replay_receipt_matches_manuscript_replay_row():
    receipt = json.loads(
        (ROOT / "paper/reproducibility/local_full_replay_audit_20261019.json").read_text()
    )
    assert receipt["status"] == "PASS_SCORE_LEVEL_REPLAY_AND_HEADLINE_COUNTS"
    assert receipt["score_rows_recomputed"] == 981
    assert receipt["specificity"] == {
        "gradable_nonreferable": 385,
        "true_negatives": 270,
        "percent": 70.12987012987013,
    }
    assert receipt["replay_missed_referable"] == 0
    assert receipt["replay_missed_ungradable"] == 0
    assert {
        source: values["cases"] for source, values in receipt["per_source_recomputed"].items()
    } == {"DDR": 625, "DRTiD": 250, "DeepDRiD": 106}

    manuscript = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    evidence_table = _table_body(manuscript, "tab:replay-model-contrast")
    assert "precomputed component scores, eye unit & Source-specific ranks, weights, and cutoffs" in evidence_table
    assert "Deterministic implementation replay only; 70.13\\% is not raw-image model performance" in evidence_table
    assert "raw UWF images, image unit" in evidence_table
    assert "Post-hoc MMRDR frontier" in evidence_table
    assert "Controlled simulation" in evidence_table


def test_selection_history_map_rules_and_both_manuscripts_agree():
    ieee = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    arxiv = (ROOT / "paper/active_retina_arxiv_full.tex").read_text()
    for source in (ieee, arxiv):
        lowered = source.lower()
        assert "selection-history evidence map" in lowered
        assert "fold-held" in lowered
        assert "exact replay" in lowered
        assert "does not adjust outcomes or provide a risk guarantee" in lowered or "does not adjust outcomes or confer a risk guarantee" in lowered
        assert "precomputed component scores, eye unit" in source
        assert "generated score channels, case unit" in source
    assert "Two positive test images have exact model-fit copies" in ieee
    assert "Two positive test images have exact model-fit copies" in arxiv


def test_source_replay_table_matches_aggregate_only_local_receipt():
    receipt = json.loads(
        (ROOT / "paper/reproducibility/local_full_replay_audit_20261019.json").read_text()
    )
    table = _table_body(
        (ROOT / "paper/active_retina_ieee_so_far.tex").read_text(), "tab:source"
    )
    table_rows = {
        cells[0]: cells
        for line in table.splitlines()
        if "&" in line
        for cells in [[cell.strip().removesuffix(r"\\").strip() for cell in line.split("&")]]
        if cells[0] in receipt["per_source_recomputed"]
    }
    assert set(table_rows) == set(receipt["per_source_recomputed"])
    for source, row in receipt["per_source_recomputed"].items():
        specificity = 100 * row["true_negatives"] / row["negatives"]
        cells = table_rows[source]
        assert int(cells[1]) == row["cases"]
        assert int(cells[2]) == row["negatives"]
        assert int(cells[5]) == row["true_negatives"]
        assert int(cells[7]) == row["missed_referable"] + row["missed_ungradable"]
        assert cells[8] == f"{specificity:.2f}\\%"
    assert "Pooled & 981 & 385 & 463 & 133 & 270 & 115 & 0 & 70.13\\%" in table


def test_posthoc_frontier_tables_match_inspected_test_diagnostic():
    diagnostic = json.loads(
        (ROOT / "paper/reproducibility/mmrdr_test_score_diagnostic_20261019.json").read_text()
    )
    rows = diagnostic["sensitivity_specificity_frontier"]
    labels = (
        "100\\% (zero observed misses)", "$\\geq$99\\%", "$\\geq$98\\%",
        "$\\geq$95\\%", "$\\geq$90\\%", "$\\geq$80\\%",
    )
    for manuscript, table_label in (
        ("paper/active_retina_ieee_so_far.tex", "tab:mmrdr-posthoc-frontier"),
        ("paper/active_retina_arxiv_full.tex", "tab:mmrdr-test-frontier"),
    ):
        table = _table_body((ROOT / manuscript).read_text(), table_label)
        assert len(rows) == len(labels) == 6
        for row, label in zip(rows, labels):
            expected = (
                f"{label} & {100 * row['sensitivity']:.2f}\\%"
                f" & {100 * row['specificity']:.2f}\\%"
            )
            assert expected in table
    assert diagnostic["status"] == "POST_HOC_TEST_SCORE_DIAGNOSTIC_NOT_VALIDATION_OR_THRESHOLD_SELECTION"
    assert "No threshold is recommended" in diagnostic["scope"]


def test_manuscript_cross_references_and_bibliography_keys_resolve():
    import re

    checked = 0
    for relative_path in (
        "paper/active_retina_ieee_so_far.tex",
        "paper/active_retina_arxiv_full.tex",
    ):
        manuscript_path = ROOT / relative_path
        if not manuscript_path.exists():
            continue
        source = manuscript_path.read_text()
        checked += 1
        labels = re.findall(r"\\label\{([^}]+)\}", source)
        refs = re.findall(r"\\(?:ref|eqref|autoref|pageref)\{([^}]+)\}", source)
        bibitems = re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", source)
        cites = []
        for cited_keys in re.findall(r"\\(?:cite|citep|citet)\{([^}]+)\}", source):
            cites.extend(key.strip() for key in cited_keys.split(","))

        assert len(labels) == len(set(labels)), f"{relative_path}: duplicate labels"
        assert set(refs) <= set(labels), f"{relative_path}: unresolved references"
        assert set(labels) <= set(refs), f"{relative_path}: unreferenced labels"
        assert len(bibitems) == len(set(bibitems)), f"{relative_path}: duplicate bibliography keys"
        assert set(cites) <= set(bibitems), f"{relative_path}: unresolved citations"
        assert set(bibitems) <= set(cites), f"{relative_path}: uncited bibliography entries"
    assert checked >= 1


def test_expert_selector_table_matches_retained_aggregate_summaries():
    source = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    table = _table_body(source, "tab:selectors")

    replay = json.loads(
        (ROOT / "results/deployable_exposed_baseline_7013_20261013_with_components/summary.json").read_text()
    )["same_cohort_replay_metrics"]
    pairwise = json.loads(
        (ROOT / "results/expert_selector_nested_crossfit_20261012.json").read_text()
    )["pooled"]
    blend_summary = json.loads(
        (ROOT / "results/expert_selector_nested_crossfit_oof_threshold_20261012.json").read_text()
    )
    nonlinear = json.loads(
        (ROOT / "results/expert_selector_nested_crossfit_nonlinear_20261012.json").read_text()
    )["pooled"]
    oracle = json.loads(
        (ROOT / "results/expert_score_oracle_union_20261012.json").read_text()
    )["pooled"]

    def assert_row(label, true_negatives, misses):
        specificity = 100 * true_negatives / 385
        row = f"{label} & {true_negatives} & {specificity:.2f}\\% & {misses}"
        assert row in table

    assert_row(
        "Historical replay",
        replay["true_negatives"],
        replay["missed_referable"] + replay["missed_ungradable"],
    )
    assert_row("Pairwise rank selector, patient-fold", pairwise["true_negatives"], pairwise["misses"])
    assert_row("Nested regularized blend", blend_summary["pooled"]["true_negatives"], blend_summary["pooled"]["misses"])

    single_tn = sum(
        item["nested_single_expert_selector"]["true_negatives"]
        for item in blend_summary["sources"].values()
    )
    single_misses = sum(
        item["nested_single_expert_selector"]["misses"]
        for item in blend_summary["sources"].values()
    )
    assert_row("Nested single-expert route", single_tn, single_misses)

    union_tn = sum(
        item["nested_crossfit_or_union"]["true_negatives"]
        for item in blend_summary["sources"].values()
    )
    union_misses = sum(
        item["nested_crossfit_or_union"]["misses"]
        for item in blend_summary["sources"].values()
    )
    assert_row("Nested OR, internally shrunk", union_tn, union_misses)
    assert_row("Nested nonlinear selector", nonlinear["true_negatives"], nonlinear["misses"])
    assert_row(
        "Full-cohort casewise oracle union",
        oracle["true_negatives"],
        oracle["observed_missed_required_cases"],
    )

    rows = [line for line in table.splitlines() if " & " in line and "\\\\" in line]
    assert len(rows) - 1 == 7
    assert "174" not in table
    assert "153" not in table


def test_quantitative_manuscript_figures_match_aggregate_sources():
    import re

    source = (ROOT / "paper/active_retina_ieee_so_far.tex").read_text()
    run_card = json.loads(
        (ROOT / "paper/reproducibility/mmrdr_uwf_official_test_run_card.json").read_text()
    )

    def figure_body(label):
        label_pos = source.index(f"\\label{{{label}}}")
        begin = source.rfind("\\begin{figure}", 0, label_pos)
        end = source.index("\\end{figure}", label_pos)
        return source[begin:end]

    transfer = figure_body("fig:operating-transfer")
    transfer_points = [
        (float(x), int(y), tag.strip())
        for x, y, tag in re.findall(r"\(([-\d.]+),([-\d.]+)\)\s*\[([^]]+)\]", transfer)
    ]
    validation = run_card["validation_result"]
    test = run_card["official_test_result"]
    assert transfer_points == [
        (100 * validation["true_negative"] / (validation["true_negative"] + validation["false_positive"]),
         validation["false_negative"], "Validation"),
        (round(100 * test["true_negative"] / (test["true_negative"] + test["false_positive"]), 3),
         test["false_negative"], "Official test"),
    ]

    candidate_rows = list(
        csv.DictReader((ROOT / "paper/submission/candidate_source_aggregates_2026-10-15.csv").open())
    )
    candidate_figure = figure_body("fig:candidate-source")
    plotted = re.findall(
        r"\\addplot\[.*?\] coordinates \{(.*?)\};\s*\\addlegendentry\{(.*?)\}",
        candidate_figure,
        flags=re.DOTALL,
    )
    assert len(plotted) == 2
    for coordinates, legend in plotted:
        panel = legend.strip()
        point_map = {
            source_name: float(percent)
            for source_name, percent in re.findall(r"\(([A-Za-z]+),([\d.]+)\)", coordinates)
        }
        rows = [row for row in candidate_rows if row["candidate_panel"] == panel]
        expected = {row["source"]: float(row["specificity_percent"]) for row in rows}
        assert point_map == expected
        assert all(
            float(row["specificity_percent"])
            == pytest.approx(100 * int(row["true_negatives"]) / int(row["negative_images"]), abs=0.0051)
            for row in rows
        )
    for row in candidate_rows:
        source_path = ROOT / row["source_artifact"]
        assert re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"])
        if source_path.is_file():
            expected_source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            assert expected_source_hash == row["source_sha256"]
    expected_misses = "DDR 1/2, DeepDRiD 0/1, and DRTiD 1/0"
    assert expected_misses in candidate_figure

    seed_rows = list(
        csv.DictReader((ROOT / "paper/submission/candidate_panel_seed_summary_2026-10-16.csv").open())
    )
    seed_figure = figure_body("fig:panel-seed-sensitivity")
    seed_plots = re.findall(
        r"\\addplot\[.*?\] coordinates \{(.*?)\};\s*\\addlegendentry\{(.*?)\}",
        seed_figure,
        flags=re.DOTALL,
    )
    seed_panel_for_legend = {
        "Oracle shortlist": "label_screened_oracle_contributors",
        "All aligned": "all_complete_aligned",
    }
    assert len(seed_plots) == 2
    for coordinates, legend in seed_plots:
        panel = seed_panel_for_legend[legend.strip()]
        plotted_seeds = {
            int(seed.removeprefix("seed")): float(percent)
            for seed, percent in re.findall(r"\((seed\d+),([\d.]+)\)", coordinates)
        }
        expected_seeds = {
            int(row["seed"]) % 100: round(float(row["specificity_percent"]), 2)
            for row in seed_rows
            if row["candidate_panel"] == panel
        }
        assert plotted_seeds == expected_seeds

    cv_table = _table_body(source, "tab:cv")
    table_points = {}
    for line in cv_table.splitlines():
        cells = [cell.strip() for cell in line.strip().rstrip("\\").split("&")]
        if len(cells) == 5 and cells[1].isdigit() and cells[3].isdigit():
            table_points[(float(cells[2].replace("\\%", "")), int(cells[3]))] = cells[0]
    procedure_figure = figure_body("fig:procedure")
    procedure_points = {
        (float(specificity), int(misses))
        for specificity, misses in re.findall(r"\(([\d.]+),(\d+)\)\s*\[[^]]+\]", procedure_figure)
    }
    assert procedure_points == set(table_points)

    procedure = figure_body("fig:procedure")
    procedure_points = {
        (float(specificity), int(misses))
        for specificity, misses in re.findall(r"\(([\d.]+),(\d+)\)\s*\[[^]]+\]", procedure)
    }
    assert procedure_points == {(70.13, 0), (69.61, 3), (64.42, 5), (65.71, 6), (63.12, 0)}

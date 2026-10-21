import importlib.util
import json
from math import comb
from pathlib import Path
import re

import pytest


SCRIPT = Path(__file__).parents[1] / "paper/reproducibility/zero_miss_panel_simulation.py"
SPEC = importlib.util.spec_from_file_location("zero_miss_panel_simulation", SCRIPT)
SIM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIM)


def test_simulation_is_deterministic_and_labeled_synthetic():
    kwargs = {"replicates": 12, "correlations": (0.5,), "panel_sizes": (1, 4)}
    first = SIM.run_simulation(**kwargs)
    second = SIM.run_simulation(**kwargs)
    assert first == second
    assert first["status"] == "SYNTHETIC_MECHANISM_SIMULATION_NOT_CLINICAL_EVIDENCE"
    assert first["single_channel_order_statistic_reference"]["probability_zero_test_misses"] == 60 / 360
    panel_reference = first["independent_panel_order_statistic_reference"]
    assert panel_reference["probability_zero_test_misses_by_panel_size"]["1"] == pytest.approx(60 / 360)
    assert len(first["summary_rows"]) == 2
    assert all(row["development_misses_per_replicate"] == 0 for row in first["summary_rows"])
    sensitivity = first["negative_sample_sensitivity"]
    assert sensitivity["status"] == "PAIRED_SYNTHETIC_SENSITIVITY_NOT_CLINICAL_EVIDENCE"
    assert len(sensitivity["summary_rows"]) == 6
    for row in first["summary_rows"]:
        assert 0 <= row["probability_zero_test_misses"] <= 1
        lower, upper = row["probability_zero_test_misses_monte_carlo_wilson95"]
        assert 0 <= lower <= row["probability_zero_test_misses"] <= upper <= 1


def test_perfectly_correlated_panel_is_a_negative_control():
    report = SIM.run_simulation(
        replicates=20, correlations=(1.0,), panel_sizes=(1, 5, 10)
    )
    rows = report["summary_rows"]
    for key in (
        "mean_development_specificity",
        "mean_independent_test_specificity",
        "mean_independent_test_sensitivity",
        "probability_zero_test_misses",
    ):
        assert len({row[key] for row in rows}) == 1
    sensitivity_rows = report["negative_sample_sensitivity"]["summary_rows"]
    for n_negative in (50, 200, 1000):
        grouped = [row for row in sensitivity_rows if row["development_negatives"] == n_negative]
        assert len({row["mean_development_specificity"] for row in grouped}) == 1
    for key in ("mean_independent_test_specificity", "probability_zero_test_misses"):
        assert len({row[key] for row in sensitivity_rows}) == 1


def test_negative_sample_sensitivity_pairs_cases_across_selection_sample_sizes():
    report = SIM.run_simulation(
        replicates=40,
        correlations=(0.9,),
        panel_sizes=(1, 5),
        sensitivity_negative_sizes=(10, 80),
    )
    sensitivity = report["negative_sample_sensitivity"]
    assert sensitivity["development_negative_sizes"] == [10, 80]
    assert "same development-positive, test-positive, and test-negative cases" in sensitivity["design"]
    rows = sensitivity["summary_rows"]
    assert len(rows) == 4
    assert {(row["development_negatives"], row["channels_searched"]) for row in rows} == {
        (10, 1), (10, 5), (80, 1), (80, 5)
    }
    assert all(row["replicates"] == 40 for row in rows)
    contrasts = sensitivity["paired_gap_contrasts"]
    assert len(contrasts) == 2
    assert all(row["replicates"] == 40 for row in contrasts)
    assert all(row["monte_carlo_se"] >= 0 for row in contrasts)
    assert all(len(row["monte_carlo_normal95"]) == 2 for row in contrasts)


def test_independent_panel_order_statistic_reference():
    f = SIM.independent_panel_zero_miss_probability
    assert f(60, 300, 1) == pytest.approx(1 / 6)
    assert f(60, 300, 5) == pytest.approx(1 / 252)
    assert f(60, 300, 20) == pytest.approx(1.8821757952e-5)
    assert f(60, 300, 100) == pytest.approx(1.0356185894e-8)


@pytest.mark.parametrize("n_dev,n_test,channels", [(2, 3, 2), (5, 7, 4), (60, 300, 5)])
def test_panel_reference_matches_order_statistic_integral(n_dev, n_test, channels):
    # Expand (1 - v**n_dev)**(K-1) in the density integral independently.
    finite_sum = channels * n_dev * sum(
        (-1) ** j * comb(channels - 1, j) /
        (n_dev + n_test + n_dev * j)
        for j in range(channels)
    )
    assert SIM.independent_panel_zero_miss_probability(
        n_dev, n_test, channels
    ) == pytest.approx(finite_sum)


def test_latex_simulation_plot_matches_saved_results():
    root = Path(__file__).parents[1]
    source = (root / "paper/active_retina_ieee_so_far.tex").read_text()
    results = json.loads((root / "paper/reproducibility/zero_miss_panel_simulation_results.json").read_text())
    figure_start = source.index("\\label{fig:panel-simulation}")
    figure = source[source.rfind("\\begin{figure}", 0, figure_start):figure_start]
    plot_blocks = re.findall(
        r"\\addplot\[.*?\] coordinates \{(.*?)\};\s*\\addlegendentry\{(.*?)\}",
        figure,
        flags=re.DOTALL,
    )
    assert len(plot_blocks) == 4
    result_map = {
        (row["correlation"], row["channels_searched"]): row
        for row in results["summary_rows"]
    }
    for coordinates, legend in plot_blocks:
        rho = float(re.search(r"\\rho=([0-9.]+)", legend).group(1))
        plotted = [(int(x), float(y)) for x, y in re.findall(r"\((\d+),([0-9.]+)\)", coordinates)]
        assert len(plotted) == 4
        for channels, percent in plotted:
            expected = 100 * result_map[(rho, channels)]["probability_zero_test_misses"]
            assert percent == pytest.approx(expected, abs=0.0051)


def test_latex_simulation_table_matches_saved_results():
    root = Path(__file__).parents[1]
    source = (root / "paper/active_retina_ieee_so_far.tex").read_text()
    results = json.loads((root / "paper/reproducibility/zero_miss_panel_simulation_results.json").read_text())
    table_start = source.index("\\label{tab:panel-simulation}")
    table_end = source.index("\\end{table}", table_start)
    table = source[table_start:table_end]
    result_map = {
        (row["correlation"], row["channels_searched"]): row
        for row in results["summary_rows"]
    }
    rows = re.findall(
        r"^([01]\.\d)\s*&\s*(\d+)\s*&\s*([\d.]+)\\%\s*&\s*([\d.]+)\\%\s*&\s*([\d.]+)\\%",
        table,
        flags=re.MULTILINE,
    )
    assert len(rows) == 6
    for rho_text, k_text, dev_text, test_text, zero_text in rows:
        row = result_map[(float(rho_text), int(k_text))]
        assert float(dev_text) == pytest.approx(100 * row["mean_development_specificity"], abs=0.0051)
        assert float(test_text) == pytest.approx(100 * row["mean_independent_test_specificity"], abs=0.0051)
        assert float(zero_text) == pytest.approx(100 * row["probability_zero_test_misses"], abs=0.0051)


def test_negative_sample_sensitivity_tables_match_saved_paired_results():
    root = Path(__file__).parents[1]
    results = json.loads((root / "paper/reproducibility/zero_miss_panel_simulation_results.json").read_text())
    sensitivity_rows = results["negative_sample_sensitivity"]["summary_rows"]
    row_map = {
        (row["correlation"], row["development_negatives"]): row
        for row in sensitivity_rows
        if row["channels_searched"] == 100 and row["correlation"] in (0.0, 0.9)
    }
    pattern = re.compile(
        r"^([01]\.\d)\s*&\s*([\d,]+)\s*&\s*([\d.]+)\\%\s*&\s*([\d.]+)\\%"
        r"\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*\(([\d.]+)--([\d.]+)\)",
        flags=re.MULTILINE,
    )
    for manuscript, label in (
        ("paper/active_retina_ieee_so_far.tex", "tab:negative-sample-sensitivity"),
        ("paper/active_retina_arxiv_full.tex", "tab:arxiv-negative-sample-sensitivity"),
    ):
        source = (root / manuscript).read_text()
        start = source.index(f"\\label{{{label}}}")
        end = source.index("\\end{table*}", start)
        table_rows = pattern.findall(source[start:end])
        assert len(table_rows) == 6, manuscript
        for rho, n_negative, dev, test, gap, zero, lower, upper in table_rows:
            row = row_map[(float(rho), int(n_negative.replace(",", "")))]
            assert float(dev) == pytest.approx(100 * row["mean_development_specificity"], abs=0.011)
            assert float(test) == pytest.approx(100 * row["mean_independent_test_specificity"], abs=0.011)
            assert float(gap) == pytest.approx(
                100 * (row["mean_development_specificity"] - row["mean_independent_test_specificity"]),
                abs=0.011,
            )
            assert float(zero) == pytest.approx(100 * row["probability_zero_test_misses"], abs=0.011)
            ci = row["probability_zero_test_misses_monte_carlo_wilson95"]
            assert float(lower) == pytest.approx(100 * ci[0], abs=0.011)
            assert float(upper) == pytest.approx(100 * ci[1], abs=0.011)


def test_reported_paired_gap_contrasts_match_replicate_level_simulation():
    root = Path(__file__).parents[1]
    results = json.loads((root / "paper/reproducibility/zero_miss_panel_simulation_results.json").read_text())
    sensitivity = results["negative_sample_sensitivity"]
    contrast_map = {
        row["correlation"]: row
        for row in sensitivity["paired_gap_contrasts"]
        if row["channels_searched"] == 100
    }
    rows = {
        (row["correlation"], row["development_negatives"]): row
        for row in sensitivity["summary_rows"]
        if row["channels_searched"] == 100
    }
    for rho in (0.0, 0.9):
        contrast = contrast_map[rho]
        expected = (
            rows[(rho, 50)]["mean_development_minus_test_specificity"]
            - rows[(rho, 1000)]["mean_development_minus_test_specificity"]
        )
        assert contrast["mean_paired_gap_reduction"] == pytest.approx(expected)
        lower, upper = contrast["monte_carlo_normal95"]
        assert lower < expected < upper
    assert contrast_map[0.0]["mean_paired_gap_reduction"] != pytest.approx(
        contrast_map[0.9]["mean_paired_gap_reduction"]
    )
    for manuscript in ("paper/active_retina_ieee_so_far.tex", "paper/active_retina_arxiv_full.tex"):
        source = (root / manuscript).read_text()
        assert "7.89" in source and "7.60--8.19" in source
        assert "3.79" in source and "3.50--4.08" in source


def test_negative_sample_figures_match_simulated_gap_and_monte_carlo_intervals():
    root = Path(__file__).parents[1]
    results = json.loads((root / "paper/reproducibility/zero_miss_panel_simulation_results.json").read_text())
    result_map = {
        (row["correlation"], row["development_negatives"]): row
        for row in results["negative_sample_sensitivity"]["summary_rows"]
        if row["channels_searched"] == 100 and row["correlation"] in (0.0, 0.9)
    }
    for manuscript, label in (
        ("paper/active_retina_ieee_so_far.tex", "fig:negative-sample-sensitivity"),
        ("paper/active_retina_arxiv_full.tex", "fig:arxiv-negative-sample-sensitivity"),
    ):
        source = (root / manuscript).read_text()
        label_at = source.index(f"\\label{{{label}}}")
        figure = source[source.rfind("\\begin{figure}", 0, label_at):label_at]
        blocks = re.findall(
            r"\\addplot\+\[(.*?)\]\s*coordinates\s*\{(.*?)\};\s*\\addlegendentry\{(.*?)\}",
            figure,
            flags=re.DOTALL,
        )
        assert len(blocks) == 2, manuscript
        for _, coordinates, legend in blocks:
            rho = float(re.search(r"\\rho=([0-9.]+)", legend).group(1))
            plotted = re.findall(
                r"\((\d+),([\d.]+)\)\s*\+\-\s*\(0,([\d.]+)\)", coordinates
            )
            assert len(plotted) == 3, manuscript
            for n_negative, gap, error in plotted:
                row = result_map[(rho, int(n_negative))]
                ci = row["specificity_gap_monte_carlo_normal95"]
                assert float(gap) == pytest.approx(
                    100 * row["mean_development_minus_test_specificity"], abs=0.0001
                )
                assert float(error) == pytest.approx(50 * (ci[1] - ci[0]), abs=0.0001)

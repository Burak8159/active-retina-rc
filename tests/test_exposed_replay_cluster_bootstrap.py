import pytest
import numpy as np

from paper.reproducibility.bootstrap_exposed_replay_specificity import (
    bootstrap_specificity,
)


def test_cluster_bootstrap_uses_patient_clusters_and_emits_no_identifiers():
    scores = [
        {"eye_id": "a_l", "source": "DeepDRiD", "referable": "0", "ungradable": "0", "action": "CLEAR"},
        {"eye_id": "a_r", "source": "DeepDRiD", "referable": "0", "ungradable": "0", "action": "REFER"},
        {"eye_id": "b_l", "source": "DeepDRiD", "referable": "0", "ungradable": "0", "action": "CLEAR"},
        {"eye_id": "c_l", "source": "DDR", "referable": "0", "ungradable": "0", "action": "REFER"},
    ]
    manifest = [
        {"eye_id": "a_l", "patient_id": "p1", "source_dataset": "deepdrid_paired", "grade": "0"},
        {"eye_id": "a_r", "patient_id": "p1", "source_dataset": "deepdrid_paired", "grade": "0"},
        {"eye_id": "b_l", "patient_id": "p2", "source_dataset": "deepdrid_paired", "grade": "0"},
        {"eye_id": "c_l", "patient_id": "p3", "source_dataset": "ddr_hf_specificity300", "grade": "0"},
    ]
    result = bootstrap_specificity(scores, manifest, replicates=2000, seed=71)

    assert result["observed"] == {
        "true_negatives": 2,
        "negative_eyes": 4,
        "specificity": 0.5,
        "negative_patient_clusters": 3,
    }
    assert result["bootstrap"]["source_strata"]["deepdrid_paired"]["negative_patient_clusters"] == 2
    assert result["bootstrap"]["source_strata"]["deepdrid_paired"]["negative_eyes"] == 3
    source_ci = result["bootstrap"]["source_strata"]["deepdrid_paired"]["patient_cluster_bootstrap_percentile_95_interval"]
    assert source_ci[0] <= source_ci[1]
    assert result["bootstrap"]["percentile_95_interval"][0] <= result["bootstrap"]["percentile_95_interval"][1]
    # Independently reconstruct the pooled draws. This catches accidental reuse
    # of the later per-source-contrast quantiles in the pooled receipt fields.
    rng = np.random.default_rng(71)
    ddr = np.asarray([[0, 1]], dtype=np.int64)
    deepdrid = np.asarray([[1, 2], [1, 1]], dtype=np.int64)
    ddr_draw = ddr[rng.integers(0, len(ddr), size=(2000, len(ddr)))].sum(axis=1)
    deepdrid_draw = deepdrid[
        rng.integers(0, len(deepdrid), size=(2000, len(deepdrid)))
    ].sum(axis=1)
    pooled_draw = (ddr_draw[:, 0] + deepdrid_draw[:, 0]) / (
        ddr_draw[:, 1] + deepdrid_draw[:, 1]
    )
    expected_pooled = np.quantile(pooled_draw, [0.025, 0.5, 0.975])
    assert result["bootstrap"]["percentile_95_interval"] == pytest.approx(
        expected_pooled[[0, 2]].tolist()
    )
    assert result["bootstrap"]["median"] == pytest.approx(expected_pooled[1])
    mixture = result["source_mix_sensitivity"]
    assert mixture["observed_mix_specificity"] == 0.5
    assert mixture["equal_source_standardized_specificity"] == pytest.approx((2 / 3 + 0) / 2)
    assert mixture["range_over_all_source_mixtures_of_observed_source_rates"] == [0.0, 2 / 3]
    assert mixture["equal_source_patient_cluster_bootstrap_percentile_95_interval"][0] <= mixture["equal_source_patient_cluster_bootstrap_percentile_95_interval"][1]
    contrasts = mixture["exploratory_pairwise_source_contrasts"]
    assert set(contrasts) == {
        "ddr_hf_specificity300_minus_deepdrid_paired",
    }
    assert all(
        item["patient_cluster_bootstrap_percentile_95_interval"][0]
        <= item["patient_cluster_bootstrap_percentile_95_interval"][1]
        for item in contrasts.values()
    )
    assert "p1" not in str(result)
    assert "policy/model selection" in result["claim_boundary"]


def test_cluster_bootstrap_fails_closed_on_group_conflicts_and_duplicates():
    score = [{"eye_id": "eye", "source": "DDR", "referable": "0", "ungradable": "0", "action": "CLEAR"}]
    manifest = [
        {"eye_id": "eye", "patient_id": "p1", "source_dataset": "ddr_hf_specificity300", "grade": "0"},
        {"eye_id": "eye", "patient_id": "p2", "source_dataset": "ddr_hf_specificity300", "grade": "0"},
    ]
    with pytest.raises(ValueError, match="conflicting group"):
        bootstrap_specificity(score, manifest, replicates=10)
    with pytest.raises(ValueError, match="duplicate eye"):
        bootstrap_specificity(score * 2, manifest[:1], replicates=10)

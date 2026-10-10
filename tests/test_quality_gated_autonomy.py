import numpy as np
from pathlib import Path

import torch

from active_retina.train_quality_gated_autonomy import (
    evaluate_policy,
    hard_boundary_rank_loss,
    masked_bce_with_logits,
    select_policy,
)
from active_retina.train_mixed_view_bag_autonomy import (
    BoundaryFocusedEyeBatchSampler,
    BoundaryPairBatchSampler,
    EyeBags,
    SourceBalancedEyeBatchSampler,
    aggregate_pretrained_view_risk,
    filter_prediction_sources,
    ordinal_referable_score,
    select_manifest_sources,
    select_filtered_policy,
)
from active_retina.autonomy_extra_signal_search import select_best_validated
from active_retina.autonomy_zero_miss_90_search import select_best_validated as select_zero_miss_search_candidate
from active_retina.augmented_full_binary_gap import select_best_validated as select_full_binary_gap_candidate
from active_retina.audit_mixed_view_ordinal_head import ordinal_referable_probability
from active_retina.train_mixed_view_cumulative_ordinal import CumulativeOrdinalHead
from active_retina.audit_mixed_view_inference_fusion import make_policy_input
from active_retina.build_autonomy_route_manifest import drop_incomplete_paired_eyes


def test_policy_routes_ungradable_and_preserves_zero_miss_on_validation():
    pred = {
        "grade": np.array([0, 1, 2, 3, 5]),
        "risk": np.array([0.1, 0.2, 0.3, 0.8, 0.1]),
        "quality": np.array([0.1, 0.2, 0.1, 0.1, 0.9]),
        "quality_known": np.array([1, 1, 1, 1, 1]),
        "ungradable": np.array([0, 0, 0, 0, 1]),
    }

    policy = select_policy(pred)
    metrics = evaluate_policy(pred, policy)

    assert metrics["missed_referable_dr"] == 0
    assert metrics["missed_known_ungradable"] == 0


def test_policy_supports_cohorts_without_quality_labels():
    pred = {
        "grade": np.array([0, 1, 2, 3]),
        "risk": np.array([0.1, 0.2, 0.7, 0.9]),
        "quality": np.array([0.9, 0.8, 0.1, 0.2]),
        "quality_known": np.zeros(4, dtype=int),
        "ungradable": np.zeros(4, dtype=int),
    }

    policy = select_policy(pred)
    metrics = evaluate_policy(pred, policy)

    assert policy["quality_threshold"] == 2.0
    assert metrics["missed_referable_dr"] == 0
    assert metrics["known_ungradable"] == 0
    assert metrics["case_action_coverage"] == 1.0
    assert metrics["sensitivity_referable_dr"] == 1.0


def test_unlabeled_quality_batch_has_finite_zero_loss_and_gradient():
    logits = torch.tensor([0.2, -0.4], requires_grad=True)
    targets = torch.zeros(2)
    mask = torch.zeros(2, dtype=torch.bool)

    loss = masked_bce_with_logits(logits, targets, mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_empty_ordinal_batch_has_finite_zero_loss_and_gradient():
    logits = torch.randn(2, 4, requires_grad=True)
    targets = torch.zeros(2, 4)
    mask = torch.zeros(2, dtype=torch.bool)

    loss = masked_bce_with_logits(logits, targets, mask)
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_hard_boundary_rank_loss_pushes_referable_above_nonreferable():
    logits = torch.tensor([0.1, 0.2, -0.1, 0.0], requires_grad=True)
    grade = torch.tensor([2, 3, 0, 1])
    referable = torch.tensor([1.0, 1.0, 0.0, 0.0])
    gradable = torch.ones(4, dtype=torch.bool)

    loss = hard_boundary_rank_loss(logits, grade, referable, gradable, 0.5, 0.5, 1.5)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad[:2].sum() < 0
    assert logits.grad[2:].sum() > 0


def test_ordinal_head_referable_probability_sums_grades_two_through_four():
    logits = torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.0]])

    probability = ordinal_referable_probability(logits)

    expected = logits.softmax(dim=1)[:, 2:].sum(dim=1)
    assert torch.allclose(probability, expected)


def test_cumulative_ordinal_logits_are_ordered_and_differentiable():
    head = CumulativeOrdinalHead(4)
    features = torch.randn(3, 12, requires_grad=True)

    logits = head(features)
    loss = logits.square().mean()
    loss.backward()

    assert logits.shape == (3, 4)
    assert torch.all(torch.diff(logits, dim=1) < 0)
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_mixed_dataset_uses_eye_as_unit_and_requires_both_drtid_views():
    rows = [
        {"eye_id": "ddr:e1", "patient_id": "ddr:p1", "source_dataset": "ddr_hf_specificity300",
         "grade": 0, "referable_dr": 0, "field": "single_fundus", "image_path": "a.jpg", "image_id": "a"},
        {"eye_id": "dr:e2", "patient_id": "dr:p2", "source_dataset": "drtid_paired",
         "grade": 2, "referable_dr": 1, "field": "optic_disc", "image_path": "b.jpg", "image_id": "b"},
        {"eye_id": "dr:e2", "patient_id": "dr:p2", "source_dataset": "drtid_paired",
         "grade": 2, "referable_dr": 1, "field": "macula", "image_path": "c.jpg", "image_id": "c"},
    ]

    bags = EyeBags(rows, transform=lambda image: image, root=Path("."))

    assert len(bags) == 2
    assert sorted(len(item["views"]) for item in bags.items) == [1, 2]


def test_boundary_pair_sampler_pairs_grade_1_and_2_within_source_and_keeps_controls():
    items = []
    for source in ("ddr", "drtid"):
        for grade in (0, 1, 2, 3, 5):
            items.append({"source_dataset": source, "grade": grade})
    sampler = BoundaryPairBatchSampler(items, batches_per_epoch=10, seed=17)
    batches = list(sampler)

    assert len(batches) == 10
    for step in range(8):
        left, right = (items[index] for index in batches[step])
        assert left["source_dataset"] == right["source_dataset"]
        assert {left["grade"], right["grade"]} == {1, 2}
    quality_batch = [items[index] for index in batches[8]]
    assert {item["grade"] for item in quality_batch} == {0, 5}
    extreme_batch = [items[index] for index in batches[9]]
    assert {item["grade"] for item in extreme_batch} == {0, 3}


def test_max_view_fusion_preserves_any_view_risk_signal():
    scores = {
        "grade": np.array([1, 2]), "quality_known": np.array([1, 1]),
        "ungradable": np.array([0, 0]), "fused": np.array([0.2, 0.3]),
        "view0": np.array([0.1, 0.8]), "view1": np.array([0.7, 0.4]),
        "quality": np.array([0.0, 0.0]),
    }

    pred = make_policy_input(scores, "max_any_view")

    assert np.allclose(pred["risk"], [0.7, 0.8])


def test_boundary_focused_sampler_emits_one_eye_and_periodic_quality_controls():
    items = []
    for source in ("ddr", "drtid"):
        for grade in (0, 1, 2, 3, 5):
            items.append({"source_dataset": source, "grade": grade})
    sampler = BoundaryFocusedEyeBatchSampler(items, samples_per_epoch=10, seed=19)
    batches = list(sampler)

    assert len(batches) == 10
    assert all(len(batch) == 1 for batch in batches)
    for step in range(8):
        assert items[batches[step][0]]["grade"] in (1, 2)
    assert items[batches[8][0]]["grade"] == 5
    assert items[batches[9][0]]["grade"] in (0, 3)


def test_route_manifest_drops_only_incomplete_paired_eyes():
    rows = [
        {"source_dataset": "deepdrid_paired", "eye_id": "p1", "field": "macula"},
        {"source_dataset": "deepdrid_paired", "eye_id": "p1", "field": "optic_disc"},
        {"source_dataset": "deepdrid_paired", "eye_id": "p2", "field": "macula"},
        {"source_dataset": "ddr", "eye_id": "p3", "field": "single_fundus"},
    ]

    kept, dropped = drop_incomplete_paired_eyes(rows)

    assert [row["eye_id"] for row in kept] == ["p1", "p1", "p3"]
    assert dropped == {"deepdrid_paired": 1}


def test_source_balanced_sampler_round_robins_domains_and_retains_quality_pairs():
    items = []
    for source in ("ddr_hf_specificity300", "drtid_paired", "deepdrid_paired"):
        for grade in (0, 1, 2, 3, 5):
            if grade == 5 and source != "ddr_hf_specificity300":
                continue
            items.append({"source_dataset": source, "grade": grade})
    sampler = SourceBalancedEyeBatchSampler(items, batches_per_epoch=10, seed=23)
    batches = list(sampler)

    assert len(batches) == 10
    for step in range(8):
        left, right = (items[index] for index in batches[step])
        assert left["source_dataset"] == right["source_dataset"]
    quality_batch = [items[index] for index in batches[9]]
    assert 5 in {item["grade"] for item in quality_batch}
    assert any(item["grade"] < 5 for item in quality_batch)


def test_prediction_source_filter_keeps_only_requested_domains_and_fails_on_empty():
    scores = {
        "source_dataset": np.array(["ddr", "deepdrid", "drtid"]),
        "risk": np.array([0.1, 0.2, 0.3]),
    }

    filtered = filter_prediction_sources(scores, ["deepdrid", "drtid"])

    assert filtered["source_dataset"].tolist() == ["deepdrid", "drtid"]
    assert filtered["risk"].tolist() == [0.2, 0.3]
    try:
        filter_prediction_sources(scores, ["missing"])
    except ValueError:
        pass
    else:
        raise AssertionError("Empty source filters must fail closed")


def test_final_policy_selection_uses_only_declared_evaluation_sources():
    scores = {
        "source_dataset": np.array(["drtid", "drtid", "ddr", "ddr"]),
        "grade": np.array([0, 2, 0, 2]),
        "risk": np.array([0.1, 0.2, 0.1, 0.05]),
        "quality": np.zeros(4),
        "quality_known": np.zeros(4, dtype=int),
        "ungradable": np.zeros(4, dtype=int),
    }

    paired, policy = select_filtered_policy(scores, ["drtid"])
    all_sources = select_policy({key: scores[key] for key in ("grade", "risk", "quality", "quality_known", "ungradable")})

    assert paired["source_dataset"].tolist() == ["drtid", "drtid"]
    assert policy["risk_threshold"] == 0.2
    assert policy["specificity_gradable_nonreferable"] == 1.0
    assert all_sources["specificity_gradable_nonreferable"] == 0.0


def test_augmented_candidate_selection_ignores_test_performance():
    rows = [
        {
            "candidate": "test_winner",
            "valid_false_negative_clearances": 0,
            "valid_specificity_among_auto_decisions": 0.91,
            "valid_negative_clearance_rate": 0.25,
            "valid_coverage": 0.60,
            "test_specificity_among_auto_decisions": 1.0,
            "test_coverage": 0.90,
        },
        {
            "candidate": "validation_winner",
            "valid_false_negative_clearances": 0,
            "valid_specificity_among_auto_decisions": 0.90,
            "valid_negative_clearance_rate": 0.35,
            "valid_coverage": 0.75,
            "test_specificity_among_auto_decisions": 0.80,
            "test_coverage": 0.70,
        },
    ]

    best = select_best_validated(rows, min_spec=0.90)

    assert best["candidate"] == "validation_winner"


def test_zero_miss_search_candidate_ranks_by_validation_not_test():
    rows = [
        {
            "candidate": "test_winner",
            "valid_false_negative_clearances": 0,
            "valid_specificity_among_auto_decisions": 0.91,
            "valid_negative_clearance_rate": 0.20,
            "valid_coverage": 0.60,
            "test_specificity_among_auto_decisions": 1.0,
            "test_negative_clearance_rate": 0.50,
        },
        {
            "candidate": "validation_winner",
            "valid_false_negative_clearances": 0,
            "valid_specificity_among_auto_decisions": 0.90,
            "valid_negative_clearance_rate": 0.40,
            "valid_coverage": 0.75,
            "test_specificity_among_auto_decisions": 0.80,
            "test_negative_clearance_rate": 0.30,
        },
    ]

    best = select_zero_miss_search_candidate(rows)

    assert best["candidate"] == "validation_winner"


def test_full_binary_gap_candidate_ranks_by_validation_not_test():
    rows = [
        {"candidate": "test_winner", "valid_fn": 0, "valid_specificity": 0.30, "test_specificity": 0.90},
        {"candidate": "validation_winner", "valid_fn": 0, "valid_specificity": 0.45, "test_specificity": 0.20},
    ]

    best = select_full_binary_gap_candidate(rows)

    assert best["candidate"] == "validation_winner"


def test_manifest_source_selection_filters_without_mutating_rows():
    rows = [
        {"source_dataset": "ddr", "image_path": "a.jpg"},
        {"source_dataset": "deepdrid", "image_path": "b.jpg"},
    ]

    selected = select_manifest_sources(rows, ["deepdrid"], "train")

    assert selected == [rows[1]]
    assert len(rows) == 2


def test_pretrained_risk_aggregation_uses_available_views_only():
    logits = torch.tensor([[0.2, 1.1], [0.7, -4.0]])
    view_mask = torch.tensor([[True, True], [True, False]])

    risk = aggregate_pretrained_view_risk(logits, view_mask)

    assert torch.allclose(risk, torch.tensor([1.1, 0.7]))


def test_ordinal_referable_score_sums_grades_two_through_four():
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0]])

    risk = ordinal_referable_score(logits)

    assert risk[0] < 0.3
    assert risk[1] > 0.9

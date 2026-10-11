"""Audit conservative late fusion of paired-view model scores on development data."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from active_retina.train_mixed_view_bag_autonomy import EyeBags, MixedViewModel, ordinal_referable_score
from active_retina.train_quality_gated_autonomy import evaluate_policy, read_rows, select_policy


def collect(model: MixedViewModel, data: EyeBags, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    values: dict[str, list[Any]] = {key: [] for key in (
        "grade", "quality_known", "ungradable", "fused", "ordinal", "view0", "view1", "quality",
        "eye_id", "patient_id", "source_dataset", "image_ids",
    )}
    model.eval()
    with torch.inference_mode():
        for images, mask, grade, _referable, quality_known, ungradable, eye_id, patient_id, source, image_ids, *_aux in loader:
            images, mask = images.to(device), mask.to(device)
            grade_logits, fused_risk, fused_quality = model(images, mask)
            first_mask = torch.zeros_like(mask)
            first_mask[:, 0] = mask[:, 0]
            second_mask = torch.zeros_like(mask)
            second_mask[:, 1] = mask[:, 1]
            missing_second = ~mask[:, 1]
            second_mask[missing_second, 0] = mask[missing_second, 0]
            _, first_risk, _ = model(images, first_mask)
            _, second_risk, _ = model(images, second_mask)
            batch_values = {
                "grade": grade.numpy(), "quality_known": quality_known.numpy(),
                "ungradable": ungradable.numpy(), "fused": fused_risk.float().sigmoid().cpu().numpy(),
                "ordinal": ordinal_referable_score(grade_logits).cpu().numpy(),
                "view0": first_risk.float().sigmoid().cpu().numpy(),
                "view1": second_risk.float().sigmoid().cpu().numpy(),
                "quality": fused_quality.float().sigmoid().cpu().numpy(),
                "eye_id": eye_id, "patient_id": patient_id, "source_dataset": source, "image_ids": image_ids,
            }
            for key, batch in batch_values.items():
                values[key].extend(batch.tolist() if isinstance(batch, np.ndarray) else list(batch))
    return {key: np.asarray(value) for key, value in values.items()}


def make_policy_input(scores: dict[str, np.ndarray], rule: str) -> dict[str, np.ndarray]:
    if rule == "fused":
        risk = scores["fused"]
    elif rule == "ordinal":
        risk = scores["ordinal"]
    elif rule == "max_any_view":
        risk = np.maximum.reduce((scores["fused"], scores["view0"], scores["view1"]))
    elif rule == "mean_fused_views":
        risk = (scores["fused"] + scores["view0"] + scores["view1"]) / 3.0
    elif rule == "max_views":
        risk = np.maximum(scores["view0"], scores["view1"])
    else:
        raise ValueError(f"Unknown score rule: {rule}")
    return {
        "grade": scores["grade"].astype(int), "risk": risk.astype(float),
        "quality": scores["quality"].astype(float),
        "quality_known": scores["quality_known"].astype(int),
        "ungradable": scores["ungradable"].astype(int),
    }


def source_metrics(pred: dict[str, np.ndarray], policy: dict[str, Any], source: np.ndarray) -> dict[str, Any]:
    output = {}
    for name in sorted(set(source.tolist())):
        indices = source == name
        subset = {key: value[indices] for key, value in pred.items()}
        output[name] = evaluate_policy(subset, policy)
    return output


def source_conditioned_evaluation(
    valid_pred: dict[str, np.ndarray],
    valid_source: np.ndarray,
    test_pred: dict[str, np.ndarray],
    test_source: np.ndarray,
) -> dict[str, Any]:
    """Select source-specific policies on validation and apply them unchanged."""
    output: dict[str, Any] = {}
    sources = sorted(set(valid_source.tolist()))
    if set(sources) != set(test_source.tolist()):
        raise ValueError("Validation and test must contain the same source set")
    for name in sources:
        valid_mask = valid_source == name
        test_mask = test_source == name
        valid_subset = {key: value[valid_mask] for key, value in valid_pred.items()}
        test_subset = {key: value[test_mask] for key, value in test_pred.items()}
        policy = select_policy(valid_subset)
        output[name] = {
            "policy_selected_on_validation": policy,
            "validation_metrics": evaluate_policy(valid_subset, policy),
            "test_metrics_development_only": evaluate_policy(test_subset, policy),
        }
    return output


def select_source_experts_from_reports(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Select checkpoint/score experts on validation, then aggregate their fixed test diagnostics."""
    if not reports:
        raise ValueError("At least one candidate report is required")
    by_candidate = {
        name: report["validation_selected_rules"] for name, report in reports.items()
    }
    first_rule = next(iter(by_candidate.values()))
    sources = set(first_rule["fused"]["source_conditioned_policies_development_only"])
    for name, rules in by_candidate.items():
        candidate_sources = set(rules["fused"]["source_conditioned_policies_development_only"])
        if candidate_sources != sources:
            raise ValueError(f"Candidate {name} has a different source set")

    selected: dict[str, Any] = {}
    total = {
        "gradable_nonreferable": 0,
        "false_referrals_on_gradable_nonreferable": 0,
        "gradable_referable_dr": 0,
        "missed_referable_dr": 0,
        "known_ungradable": 0,
        "missed_known_ungradable": 0,
        "n": 0,
    }
    validation_total = {key: 0 for key in total}
    rules = sorted(next(iter(by_candidate.values())).keys())
    for source in sorted(sources):
        choices = []
        for candidate, candidate_rules in by_candidate.items():
            if set(candidate_rules) != set(rules):
                raise ValueError(f"Candidate {candidate} has a different score-rule set")
            for rule, rule_report in candidate_rules.items():
                entry = rule_report["source_conditioned_policies_development_only"][source]
                valid, test = entry["validation_metrics"], entry["test_metrics_development_only"]
                choices.append((
                    valid["missed_referable_dr"],
                    valid["missed_known_ungradable"],
                    -valid["specificity_gradable_nonreferable"],
                    valid["referral_rate"],
                    candidate,
                    rule,
                    entry,
                ))
        choice = min(choices, key=lambda item: item[:6])
        valid, test = choice[6]["validation_metrics"], choice[6]["test_metrics_development_only"]
        selected[source] = {
            "checkpoint": choice[4],
            "score_rule": choice[5],
            "validation_metrics": valid,
            "test_metrics_development_only": test,
        }
        for key in total:
            total[key] += int(test.get(key, 0))
            validation_total[key] += int(valid.get(key, 0))
    negatives = total["gradable_nonreferable"]
    referable = total["gradable_referable_dr"]
    ungradable = total["known_ungradable"]
    aggregate = {
        **total,
        "specificity_gradable_nonreferable": (
            (negatives - total["false_referrals_on_gradable_nonreferable"]) / negatives if negatives else None
        ),
        "sensitivity_referable_dr": (
            (referable - total["missed_referable_dr"]) / referable if referable else None
        ),
        "case_action_coverage": 1.0,
        "status": "EXPOSED_TEST_STABILITY_DIAGNOSTIC_ONLY",
    }
    valid_negatives = validation_total["gradable_nonreferable"]
    valid_referable = validation_total["gradable_referable_dr"]
    validation_aggregate = {
        **validation_total,
        "specificity_gradable_nonreferable": (
            (valid_negatives - validation_total["false_referrals_on_gradable_nonreferable"]) / valid_negatives
            if valid_negatives else None
        ),
        "sensitivity_referable_dr": (
            (valid_referable - validation_total["missed_referable_dr"]) / valid_referable
            if valid_referable else None
        ),
        "case_action_coverage": 1.0,
        "status": "VALIDATION_SELECTED_DEVELOPMENT_ONLY",
    }
    return {
        "selected_expert_by_source": selected,
        "aggregate_validation_selection_diagnostic": validation_aggregate,
        "aggregate_test_diagnostic": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path.cwd()
    splits = read_rows(Path(args.manifest))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    model = MixedViewModel(
        str(checkpoint["model_name"]),
        pretrained=False,
        image_size=int(checkpoint_args.get("image_size", args.image_size)),
        preserve_pretrained_risk=bool(checkpoint_args.get("preserve_pretrained_risk", False)),
        risk_pooling=str(checkpoint_args.get("risk_pooling", "fused")),
        risk_pool_temperature=float(checkpoint_args.get("risk_pool_temperature", 0.25)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    data = {
        split: EyeBags(splits[split], model.eval_transform, root)
        for split in ("valid", "test")
    }
    scores = {split: collect(model, dataset, device, args.batch_size) for split, dataset in data.items()}

    rules = ("ordinal", "fused", "max_views", "max_any_view", "mean_fused_views")
    report: dict[str, Any] = {
        "status": "DEVELOPMENT_ONLY_NOT_CERTIFICATION",
        "claim_boundary": "This manifest informed prior training and model selection; all results are development evidence.",
        "unit_of_analysis": "one DDR eye or paired DRTiD eye",
        "checkpoint": args.checkpoint,
        "model_configuration": {
            "image_size": int(checkpoint_args.get("image_size", args.image_size)),
            "risk_pooling": model.risk_pooling,
            "risk_pool_temperature": model.risk_pool_temperature,
        },
        "validation_selected_rules": {},
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for rule in rules:
        val = make_policy_input(scores["valid"], rule)
        policy = select_policy(val)
        report["validation_selected_rules"][rule] = {
            "policy": policy,
            "validation_metrics": evaluate_policy(val, policy),
            "test_metrics_development_only": evaluate_policy(make_policy_input(scores["test"], rule), policy),
            "validation_source_metrics": source_metrics(val, policy, scores["valid"]["source_dataset"]),
            "test_source_metrics_development_only": source_metrics(
                make_policy_input(scores["test"], rule), policy, scores["test"]["source_dataset"]
            ),
            "source_conditioned_policies_development_only": source_conditioned_evaluation(
                make_policy_input(scores["valid"], rule),
                scores["valid"]["source_dataset"],
                make_policy_input(scores["test"], rule),
                scores["test"]["source_dataset"],
            ),
        }
        rows.append({
            "rule": rule,
            "validation_specificity": policy["specificity_gradable_nonreferable"],
            "validation_misses": policy["missed_referable_dr"],
            "validation_referral_rate": policy["referral_rate"],
            **{
                f"test_{key}": value
                for key, value in report["validation_selected_rules"][rule]["test_metrics_development_only"].items()
            },
        })
    report["best_validation_rule"] = max(
        rules,
        key=lambda rule: (
            report["validation_selected_rules"][rule]["validation_metrics"]["specificity_gradable_nonreferable"],
            -report["validation_selected_rules"][rule]["validation_metrics"]["referral_rate"],
        ),
    )
    with (output / "fusion_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

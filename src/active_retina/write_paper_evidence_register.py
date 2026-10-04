"""Write a canonical evidence register for the ActiveRetina paper.

This keeps the manuscript honest: every central claim points to a frozen result
file, a concrete endpoint, and a GO/NO-GO state.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


OUT_DIR = Path("results/paper_evidence_register")


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"status": "MISSING", "path": str(p)}
    return json.loads(p.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pct(x: Any) -> str:
    if x is None or x == "":
        return "NA"
    try:
        return f"{100 * float(x):.1f}%"
    except (TypeError, ValueError):
        return str(x)


def num(x: Any) -> str:
    if x is None or x == "":
        return "NA"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def summarize() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selective = load_json("results/autonomy_zero_miss_90_search_summary.json")
    full_cov = load_json("results/autonomy_full_coverage_meta_search_summary.json")
    full_gap = load_json("results/full_autonomy_gap_summary.json")
    hard_resnet = load_json("results/clearance_ranker_hardmined_v2_local/summary.json")
    hard_ensemble = load_json("results/clearance_ranker_hardmined_v2_ensemble_local/summary.json")
    drtid_manifest = load_json("results/drtid_untouched_certification/untouched_validation_summary.json")
    drtid_cert = load_json("results/drtid_untouched_certification/certification_summary.json")
    drtid_recovery = load_json("results/drtid_recovery_model_search/summary.json")
    model_zoo = load_json("results/ddr_model_zoo/summary.json")
    three_output = load_json("results/three_output_autonomous_triage/summary.json")

    sel_best = selective.get("best", {})
    full_best = full_cov.get("best", {})
    hard_resnet_test = hard_resnet.get("test_metrics", {})
    hard_ensemble_test = hard_ensemble.get("test_metrics", {})
    drtid_patient = drtid_cert.get("patient_certificate", {})
    drtid_recovery_best = drtid_recovery.get("best_candidate", {})
    three_test = three_output.get("test_triage_metrics", {})

    rows = [
        {
            "claim": "Every input can receive a completed autonomous product output using a three-output triage contract.",
            "status": three_output.get("status"),
            "dataset": "DDR public subset",
            "evidence_file": "results/three_output_autonomous_triage/summary.json",
            "primary_endpoint": "autonomous output for every image, zero false-negative clearances, and >=90% specificity among diagnostic decisions",
            "key_result": (
                f"autonomous output rate={pct(three_test.get('autonomous_output_rate'))}; "
                f"false-negative clearances={three_test.get('false_negative_clearances')}; "
                f"diagnostic-decision specificity={pct(three_test.get('specificity_among_diagnostic_decisions'))}; "
                f"diagnostic decision rate={pct(three_test.get('diagnostic_decision_rate'))}; "
                f"insufficient-evidence rate={pct(three_test.get('insufficient_evidence_rate'))}"
            ),
            "paper_use": "Positive product-output result; insufficient evidence is a completed autonomous workflow action, not a binary diagnosis.",
        },
        {
            "claim": "Selective single-image autonomy can meet zero-miss and >=90% specificity among automatic decisions on DDR.",
            "status": selective.get("status"),
            "dataset": "DDR public subset",
            "evidence_file": "results/autonomy_zero_miss_90_search_summary.json",
            "primary_endpoint": "zero false-negative clearances and >=90% specificity among automatic decisions",
            "key_result": (
                f"test FN clearances={sel_best.get('test_false_negative_clearances')}; "
                f"automatic-decision specificity={pct(sel_best.get('test_specificity_among_auto_decisions'))}; "
                f"coverage={pct(sel_best.get('test_coverage'))}"
            ),
            "paper_use": "Positive result for risk-controlled selective autonomy.",
        },
        {
            "claim": "Existing score/meta-model families do not achieve every-image full autonomous diagnosis.",
            "status": full_cov.get("status"),
            "dataset": "DDR public subset",
            "evidence_file": "results/autonomy_full_coverage_meta_search_summary.json",
            "primary_endpoint": "zero false negatives and >=90% specificity with coverage=100%",
            "key_result": (
                f"best full-coverage test FN={full_best.get('test_fn')}; "
                f"specificity={pct(full_best.get('test_specificity'))}; "
                f"AUROC={num(full_best.get('test_auroc'))}"
            ),
            "paper_use": "Negative falsification result: high AUROC is insufficient for full autonomy.",
        },
        {
            "claim": "Even an optimistic test-label oracle over the current score family does not reach full-autonomy 90/0.",
            "status": full_gap.get("status"),
            "dataset": "DDR public subset",
            "evidence_file": "results/full_autonomy_gap_summary.json",
            "primary_endpoint": "best possible zero-miss threshold over current score family",
            "key_result": (
                f"best oracle specificity={pct(full_gap.get('best_oracle_candidate', {}).get('specificity'))}; "
                f"distance to 90%={pct(full_gap.get('best_oracle_candidate', {}).get('distance_to_90_specificity_points'))}"
            ),
            "paper_use": "Mechanistic evidence that the current representation family is the bottleneck.",
        },
        {
            "claim": "Hard-mined ResNet-feature clearance ranker does not solve the 90/0 every-image clearance problem.",
            "status": hard_resnet.get("status"),
            "dataset": "DDR public subset, ResNet feature cache",
            "evidence_file": "results/clearance_ranker_hardmined_v2_local/summary.json",
            "primary_endpoint": "zero false nonreferable clearance and >=90% nonreferable clearance specificity",
            "key_result": (
                f"test false clearances={hard_resnet_test.get('false_nonreferable_clearance')}; "
                f"nonreferable clearance specificity={pct(hard_resnet_test.get('nonreferable_clearance_specificity'))}"
            ),
            "paper_use": "Negative result for cached ResNet features.",
        },
        {
            "claim": "Hard-mined ensemble-feature clearance ranker is safe but does not solve the 90/0 every-image clearance problem.",
            "status": hard_ensemble.get("status"),
            "dataset": "DDR public subset, ensemble feature cache",
            "evidence_file": "results/clearance_ranker_hardmined_v2_ensemble_local/summary.json",
            "primary_endpoint": "zero false nonreferable clearance and >=90% nonreferable clearance specificity",
            "key_result": (
                f"test false clearances={hard_ensemble_test.get('false_nonreferable_clearance')}; "
                f"nonreferable clearance specificity={pct(hard_ensemble_test.get('nonreferable_clearance_specificity'))}"
            ),
            "paper_use": "Safe but low-coverage clearance; supports selective, not every-image, autonomy.",
        },
        {
            "claim": "DRTiD provides enough untouched positive cases for external certification.",
            "status": drtid_manifest.get("status"),
            "dataset": "DRTiD",
            "evidence_file": "results/drtid_untouched_certification/untouched_validation_summary.json",
            "primary_endpoint": ">=59 positive patients, preferably >=80, with complete paired images",
            "key_result": (
                f"patients={drtid_manifest.get('n_patients')}; "
                f"positive patients={drtid_manifest.get('positive_patients')}; "
                f"complete paired eyes={drtid_manifest.get('n_complete_paired_eyes')}"
            ),
            "paper_use": "External cohort eligibility, separate from model success.",
        },
        {
            "claim": "Frozen development artifact fails DRTiD untouched certification, revealing domain shift.",
            "status": drtid_cert.get("decision", drtid_cert.get("status")),
            "dataset": "DRTiD",
            "evidence_file": "results/drtid_untouched_certification/certification_summary.json",
            "primary_endpoint": "patient-level zero misses and finite-sample 5% missed-risk certificate",
            "key_result": (
                f"missed positive patients={drtid_patient.get('missed_positive_patients')}; "
                f"specificity among decided negatives={pct(drtid_patient.get('specificity_decided_negative_patients'))}; "
                f"missed-risk UCB95={pct(drtid_patient.get('patient_missed_risk_ucb_95'))}"
            ),
            "paper_use": "External failure/domain-shift evidence.",
        },
        {
            "claim": "DRTiD domain-specific recovery can produce a useful zero-miss mode.",
            "status": drtid_recovery.get("status"),
            "dataset": "DRTiD inspected recovery split",
            "evidence_file": "results/drtid_recovery_model_search/summary.json",
            "primary_endpoint": "zero misses with >=59 decided positive patients and useful specificity",
            "key_result": (
                f"candidate={drtid_recovery_best.get('candidate')}; "
                f"zero-miss decided positives={drtid_recovery_best.get('best_useful_zero_miss_59_decided_positive_patients')}; "
                f"specificity={pct(drtid_recovery_best.get('best_useful_zero_miss_59_specificity'))}; "
                f"UCB95={pct(drtid_recovery_best.get('best_certified_ucb95'))}"
            ),
            "paper_use": "Development-only recovery evidence; not final certificate.",
        },
        {
            "claim": "The model comparison package is generated and reproducible.",
            "status": model_zoo.get("status"),
            "dataset": "DDR public subset",
            "evidence_file": "results/ddr_model_zoo/summary.json",
            "primary_endpoint": "model-zoo table and figures exist",
            "key_result": f"n_models={model_zoo.get('n_models')}; metrics={model_zoo.get('metrics')}",
            "paper_use": "Paper-ready model comparison block.",
        },
    ]

    summary = {
        "status": "PAPER_EVIDENCE_REGISTER_READY",
        "decision": "GO_THREE_OUTPUT_AUTONOMOUS_TRIAGE__NO_GO_EVERY_IMAGE_BINARY_DIAGNOSIS",
        "central_supported_claim": (
            "Every-input autonomous triage is supported on DDR when insufficient-evidence/reacquire is an explicit output; every-image binary autonomous diagnosis is not supported by current score, meta-model, or cached-feature families."
        ),
        "next_required_model_gate": (
            "Train a representation-changing image-level model and pass zero validation misses with >=90% specificity before one-shot test."
        ),
        "n_evidence_items": len(rows),
    }
    return rows, summary


def write_markdown(rows: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# ActiveRetina Paper Evidence Register\n\n",
        f"Decision: `{summary['decision']}`\n\n",
        f"Central supported claim: {summary['central_supported_claim']}\n\n",
        f"Next required model gate: {summary['next_required_model_gate']}\n\n",
        "## Evidence Items\n\n",
    ]
    for i, row in enumerate(rows, 1):
        lines.extend(
            [
                f"### E{i}. {row['claim']}\n\n",
                f"- Status: `{row['status']}`\n",
                f"- Dataset: {row['dataset']}\n",
                f"- Evidence file: `{row['evidence_file']}`\n",
                f"- Primary endpoint: {row['primary_endpoint']}\n",
                f"- Key result: {row['key_result']}\n",
                f"- Paper use: {row['paper_use']}\n\n",
            ]
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, summary = summarize()
    write_csv(OUT_DIR / "evidence_register.csv", rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(rows, summary, OUT_DIR / "evidence_register.md")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

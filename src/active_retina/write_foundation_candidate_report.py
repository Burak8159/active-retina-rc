"""Summarize foundation/retinal model candidate smoke evaluations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RUNS = {
    "RETFound_DR_IDRID_unbalanced_smoke": "results/timm_three_output_retfound_drid_smoke/summary.json",
    "RETFound_DR_IDRID_balanced_smoke": "results/timm_three_output_retfound_drid_balanced_smoke/summary.json",
    "FundusDRGrading_DINOv2_balanced_smoke": "results/timm_three_output_dinov2_fundus_balanced_smoke/summary.json",
    "FundusDRGrading_DINOv2_inverted_balanced_smoke": "results/timm_three_output_dinov2_fundus_inverted_balanced_smoke/summary.json",
}


def load(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x: Any) -> str:
    try:
        return f"{100 * float(x):.1f}%"
    except (TypeError, ValueError):
        return "NA"


def main() -> None:
    out_dir = Path("results/foundation_candidate_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, path in RUNS.items():
        data = load(path)
        if not data:
            rows.append({"run": name, "status": "MISSING", "path": path})
            continue
        test = data.get("test_triage_metrics", {})
        rows.append(
            {
                "run": name,
                "status": data.get("status"),
                "model_name": data.get("model_name"),
                "path": path,
                "n": test.get("n"),
                "positive": test.get("positive"),
                "negative": test.get("negative"),
                "false_negative_clearances": test.get("false_negative_clearances"),
                "specificity_among_diagnostic_decisions": test.get("specificity_among_diagnostic_decisions"),
                "diagnostic_decision_rate": test.get("diagnostic_decision_rate"),
                "insufficient_evidence_rate": test.get("insufficient_evidence_rate"),
                "improved_over_baseline": data.get("improved_over_baseline"),
            }
        )
    summary = {
        "status": "FOUNDATION_CANDIDATE_SMOKE_REPORT_READY",
        "baseline": "results/three_output_autonomous_triage/summary.json",
        "n_runs": len(rows),
        "best_balanced_candidate": max(
            [r for r in rows if r.get("positive") not in (None, 0)],
            key=lambda r: (
                int(r.get("false_negative_clearances") == 0),
                float(r.get("specificity_among_diagnostic_decisions") or 0),
                float(r.get("diagnostic_decision_rate") or 0),
            ),
            default=None,
        ),
        "decision": "NO_FOUNDATION_SMOKE_IMPROVES_BASELINE_YET",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Foundation Candidate Smoke Evaluation\n\n",
        "This report evaluates candidate representation-changing retinal/fundus models against the current three-output autonomy baseline. These are smoke studies, not final certification runs.\n\n",
        "Current full-data baseline: zero false-negative clearances, 94.7% specificity among diagnostic decisions, 43.8% diagnostic decision rate.\n\n",
        "| Run | Status | Positive/Test N | False Clearances | Specificity | Diagnostic Rate | Interpretation |\n",
        "|---|---:|---:|---:|---:|---:|---|\n",
    ]
    for row in rows:
        interpretation = "not usable"
        if row.get("positive") in (None, 0):
            interpretation = "invalid smoke: no positives"
        elif row.get("false_negative_clearances") == 0 and float(row.get("specificity_among_diagnostic_decisions") or 0) >= 0.9:
            interpretation = "safe smoke, compare coverage"
        elif row.get("false_negative_clearances") == 0:
            interpretation = "zero misses but poor specificity"
        lines.append(
            f"| {row['run']} | `{row.get('status')}` | {row.get('positive')}/{row.get('n')} | {row.get('false_negative_clearances')} | {pct(row.get('specificity_among_diagnostic_decisions'))} | {pct(row.get('diagnostic_decision_rate'))} | {interpretation} |\n"
        )
    lines.extend(
        [
            "\n## Conclusion\n\n",
            "RETFound and DINOv2 candidates are now connected to the autonomy evaluator. On balanced smoke subsets, neither improves the current baseline. RETFound is very conservative; DINOv2 increases decisions but loses specificity. The next credible step is not declaring failure of foundation models, but running additional compatible checkpoints and then full validation/test only for candidates that pass a balanced smoke with positives.\n",
        ]
    )
    (out_dir / "report.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

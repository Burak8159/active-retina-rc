"""Summarize the remaining gap after the accepted augmented autonomy mode."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["referable_dr"] = int(row["referable_dr"])
        row["score"] = float(row["score"])
        row["low_threshold"] = float(row["low_threshold"])
        row["high_threshold"] = float(row["high_threshold"])
    return rows


def pct(num: int | float, den: int | float) -> str:
    if den == 0:
        return "NA"
    return f"{100.0 * float(num) / float(den):.1f}%"


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == split]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter((row["referable_dr"], row["action"]) for row in rows)
    action_counts = Counter(row["action"] for row in rows)
    positive = sum(row["referable_dr"] == 1 for row in rows)
    negative = sum(row["referable_dr"] == 0 for row in rows)
    clear_pos = counts[(1, "AUTONOMOUS_NONREFERABLE")]
    clear_neg = counts[(0, "AUTONOMOUS_NONREFERABLE")]
    refer_pos = counts[(1, "AUTONOMOUS_REFERABLE")]
    refer_neg = counts[(0, "AUTONOMOUS_REFERABLE")]
    abstain_pos = counts[(1, "AUTONOMOUS_INSUFFICIENT_EVIDENCE_REACQUIRE")]
    abstain_neg = counts[(0, "AUTONOMOUS_INSUFFICIENT_EVIDENCE_REACQUIRE")]
    diagnostic = clear_pos + clear_neg + refer_pos + refer_neg
    return {
        "n": len(rows),
        "positive": positive,
        "negative": negative,
        "clear_positive": clear_pos,
        "clear_negative": clear_neg,
        "refer_positive": refer_pos,
        "refer_negative": refer_neg,
        "abstain_positive": abstain_pos,
        "abstain_negative": abstain_neg,
        "diagnostic_decisions": diagnostic,
        "abstentions": action_counts["AUTONOMOUS_INSUFFICIENT_EVIDENCE_REACQUIRE"],
        "diagnostic_coverage": pct(diagnostic, len(rows)),
        "abstention_rate": pct(action_counts["AUTONOMOUS_INSUFFICIENT_EVIDENCE_REACQUIRE"], len(rows)),
        "negative_clearance_rate": pct(clear_neg, negative),
        "referable_referral_rate": pct(refer_pos, positive),
        "false_negative_clearances": clear_pos,
        "false_positive_referrals": refer_neg,
        "specificity_among_diagnostic_decisions": pct(clear_neg, clear_neg + refer_neg),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="results/autonomy_extra_signal_clementp_efficientnet_b5_random5000_fast/best_predictions.csv")
    parser.add_argument("--output-dir", default="results/augmented_autonomy_gap_analysis")
    args = parser.parse_args()

    rows = read_rows(Path(args.predictions))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    test = split_rows(rows, "test")
    valid = split_rows(rows, "valid")
    abstained_test = [row for row in test if row["action"] == "AUTONOMOUS_INSUFFICIENT_EVIDENCE_REACQUIRE"]
    false_positive_referrals = [
        row for row in test if row["referable_dr"] == 0 and row["action"] == "AUTONOMOUS_REFERABLE"
    ]
    write_csv(out / "test_abstentions.csv", abstained_test)
    write_csv(out / "test_false_positive_referrals.csv", false_positive_referrals)

    valid_summary = summarize(valid)
    test_summary = summarize(test)
    lines = [
        "# Augmented Autonomy Gap Analysis",
        "",
        "Accepted policy: `ActiveRetina-DDR-augmented-selective-fusion-v2`.",
        "",
        "## Validation Split",
        "",
    ]
    for key, value in valid_summary.items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Held-Out Test Split", ""]
    for key, value in test_summary.items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## What Still Blocks Every-Image Binary Autonomy",
        "",
        f"- The accepted policy abstains on `{test_summary['abstentions']}` of `{test_summary['n']}` held-out test images.",
        f"- Abstentions include `{test_summary['abstain_positive']}` referable and `{test_summary['abstain_negative']}` nonreferable images.",
        f"- The policy falsely refers `{test_summary['false_positive_referrals']}` nonreferable images among automatic decisions.",
        "- Therefore, the next model improvement target is not lower missed-disease risk inside the accepted selective mode; it is separating abstained/false-positive nonreferable images from hard referable images without introducing any false-negative clearances.",
        "",
        "Case lists:",
        "",
        "- `test_abstentions.csv`",
        "- `test_false_positive_referrals.csv`",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(out / "report.md")


if __name__ == "__main__":
    main()

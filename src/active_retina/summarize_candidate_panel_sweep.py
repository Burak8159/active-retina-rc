"""Summarize paired patient-fold seed sensitivity for candidate panels."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20261015, 20261016, 20261017, 20261018, 20261019)


def read_result(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    pooled = data["pooled"]
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "candidate_mode": data["candidate_mode"],
        "specificity": float(pooled["specificity"]),
        "true_negatives": int(pooled["true_negatives"]),
        "negatives": int(pooled["negatives"]),
        "misses": int(pooled["misses"]),
        "cases": int(pooled["cases"]),
        "patients": int(pooled["unique_patients"]),
    }


def summarize(results_dir: Path) -> dict:
    paired = []
    for seed in SEEDS:
        if seed == 20261015:
            paths = {
                "oracle_contributors": results_dir / "candidate_set_contamination_oracle_shortlist_20261015.json",
                "all_aligned": results_dir / "candidate_set_contamination_all_aligned_20261015.json",
            }
        else:
            paths = {
                mode: results_dir / f"candidate_panel_{mode}_{seed}.json"
                for mode in ("oracle_contributors", "all_aligned")
            }
        rows = {mode: read_result(path) for mode, path in paths.items()}
        if any(row["candidate_mode"] != mode for mode, row in rows.items()):
            raise ValueError(f"Candidate mode mismatch for seed {seed}")
        if len({(r["cases"], r["patients"], r["negatives"]) for r in rows.values()}) != 1:
            raise ValueError(f"Denominators differ between panels for seed {seed}")
        paired.append({"seed": seed, **rows,
                       "specificity_difference_pp_all_minus_oracle":
                           100 * (rows["all_aligned"]["specificity"] - rows["oracle_contributors"]["specificity"])})

    summary = {}
    for mode in ("oracle_contributors", "all_aligned"):
        rows = [pair[mode] for pair in paired]
        points = [100 * row["specificity"] for row in rows]
        summary[mode] = {
            "runs": len(rows),
            "mean_specificity_percent": statistics.mean(points),
            "sd_across_seeds_percentage_points": statistics.stdev(points),
            "range_specificity_percent": [min(points), max(points)],
            "mean_true_negatives": statistics.mean(row["true_negatives"] for row in rows),
            "mean_referable_misses": statistics.mean(row["misses"] for row in rows),
            "zero_miss_runs": sum(row["misses"] == 0 for row in rows),
        }
    differences = [row["specificity_difference_pp_all_minus_oracle"] for row in paired]
    summary["paired_panel_contrast"] = {
        "mean_difference_percentage_points": statistics.mean(differences),
        "range_percentage_points": [min(differences), max(differences)],
        "all_aligned_lower_runs": sum(value < 0 for value in differences),
        "all_aligned_higher_runs": sum(value > 0 for value in differences),
    }
    return {
        "status": "EXPOSED_COHORT_SPLIT_SEED_SENSITIVITY_ONLY",
        "protocol": "five-outer/four-inner patient-grouped selection; paired candidate panels; fixed integer RNG seeds 20261015-20261019; first is original split and four additional seeds assess split sensitivity",
        "cohort": {"cases": 981, "patients": 928, "negative_eyes": 385},
        "interpretation": "Seed variation is descriptive, not an independent replication or confidence interval. The cohort and artifact library were historically exposed; neither panel achieved zero misses in any run.",
        "summary": summary,
        "paired_runs": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/candidate_panel_seed_sweep_20261015.json")
    args = parser.parse_args()
    report = summarize(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

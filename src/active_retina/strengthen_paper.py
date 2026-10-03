"""Reproduce uncertainty and operational comparisons from frozen development outputs."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .write_fast_paper_pack import (
    BRANCHES, confidence_triage_metrics, pair_predictions, read_predictions,
    two_stage_metrics, write_csv, write_json,
)


def interval(values):
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return {"low": None, "high": None, "valid_replicates": 0}
    return {"low": values[int(.025 * (len(values) - 1))],
            "high": values[int(.975 * (len(values) - 1))],
            "valid_replicates": len(values)}


def main():
    source = Path("results/fast_paper_evidence_pack.json")
    pack = json.loads(source.read_text())
    rng = random.Random(20261003)
    outputs, table, provenance = [], [], {}
    for branch, report in zip(BRANCHES, pack["branches"]):
        path = branch["predictions"]
        provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        examples = [x for x in pair_predictions(read_predictions(path), branch["first_view"], branch["second_view"]).values() if x["split"] == "validation"]
        groups = defaultdict(list)
        for x in examples:
            groups[x["patient_id"]].append(x)
        ids = sorted(groups)
        active = report["baselines"]["active_retina_rc_two_stage"]["thresholds"]
        triage = report["baselines"]["confidence_triage_no_acquisition"]
        def evaluate(xs):
            a = two_stage_metrics(xs, **active)
            b = confidence_triage_metrics(xs, triage["stop_low"], triage["stop_high"])
            return a, b
        observed_a, observed_b = evaluate(examples)
        draws = defaultdict(list)
        for _ in range(1000):
            sample = []
            for occurrence in range(len(ids)):
                selected = rng.choice(ids)
                # Distinct bootstrap copies must remain distinct patient clusters.
                sample.extend({**x, "patient_id": str(occurrence)} for x in groups[selected])
            a, b = evaluate(sample)
            for metric in ("mean_images", "defer_rate", "missed_referable_risk_decided", "specificity_decided", "patient_missed_risk_decided_positive"):
                draws["active_" + metric].append(a[metric])
                draws["triage_" + metric].append(b[metric])
                draws["delta_" + metric].append(a[metric] - b[metric])
        intervals = {k: interval(v) for k, v in draws.items()}
        cost_rows = []
        for defer_cost in (0, .5, 1, 1.5, 2, 3, 5, 10):
            delta = observed_a["mean_images"] - observed_b["mean_images"] + defer_cost * (observed_a["defer_rate"] - observed_b["defer_rate"])
            ci = interval([i + defer_cost * d for i, d in zip(draws["delta_mean_images"], draws["delta_defer_rate"])])
            cost_rows.append({"branch": branch["name"], "defer_cost_images": defer_cost, "delta_cost": delta, "ci_low": ci["low"], "ci_high": ci["high"]})
        table.extend(cost_rows)
        outputs.append({"branch": branch["name"], "patients": len(ids), "eyes": len(examples), "active": observed_a, "triage": observed_b, "bootstrap_ci": intervals})
    write_json(Path("results/stronger_plan_uncertainty.json"), {"decision": "GO-ARXIV / CONDITIONAL-IEEE", "seed": 20261003, "replicates": 1000, "interpretation": "Exploratory fixed-policy patient cluster bootstrap; excludes selection uncertainty and cannot certify safety or noninferiority.", "sha256": provenance, "branches": outputs})
    write_csv(Path("results/table_operational_cost_sensitivity.csv"), table)
    manuscript = Path("paper/active_retina_rc_fast_draft.tex")
    text = manuscript.read_text()
    if r"\usepackage{amssymb}" not in text:
        text = text.replace(r"\usepackage{amsmath}", r"\usepackage{amsmath}" + "\n" + r"\usepackage{amssymb}")
    text = text.replace("Risk-Controlled Adaptive Retinal Imaging With Automatic Resolution", "Adaptive Retinal Imaging With Explicit Risk and Deferral Accounting")
    text = text.replace("risk-controlled adaptive imaging formulation", "risk-constrained development formulation")
    text = text.replace("matched observed safety", "equal observed missed-DR point estimates")
    text = text.replace("If the two-view score remains below the final referral threshold, the eye is deferred rather than automatically ruled negative.", "The upgraded policy uses separate lower and upper thresholds after acquisition: it stops negative below the lower threshold, stops positive above the upper threshold, and defers between them. The legacy policy is retained as an ablation.")
    text = text.replace("This is favorable whenever avoiding one deferred case is worth more than approximately 1.2--1.4 extra images operationally.", "Under an illustrative cost model that counts images and deferred cases only, this lowers cost when deferral costs exceed approximately 1.2--1.4 image-equivalents. This calculation excludes false-positive referral costs and does not establish clinical cost-effectiveness.")
    insertion = r"""
\section{Operational Objective and Statistical Interpretation}
Let $I$ be images acquired, $D$ indicate deferral, and $Y$ denote referable DR. We compare $C_\lambda=\mathbb{E}[I]+\lambda\mathbb{E}[D]$, with $\lambda$ an explicitly hypothetical deferral cost in image-equivalents. For an acquisition policy $a$ and triage policy $t$, the break-even value is $(\mathbb{E}[I_a]-\mathbb{E}[I_t])/(\mathbb{E}[D_t]-\mathbb{E}[D_a])$ when the denominator is positive. This algebraic comparison is meaningful only together with error, specificity, and coverage reporting. A clinical objective would additionally account for false-positive referrals and missed disease; those costs were not measured here.

The validation cohort contains 78 eyes and 32 referable eyes. Both upgraded branches miss two referable eyes (6.25\%), exceeding the 5\% calibration target. The suffix1-initial policy also attains specificity below the calibration constraint (0.457 versus 0.50). Therefore neither calibration feasibility nor equal observed missed-disease rates demonstrates validation safety, equivalence, or noninferiority. The corresponding patient missed-disease fraction is 2/18. Zero observed deferrals does not establish zero population deferral probability.

Uncertainty is estimated by 1,000 paired patient-cluster bootstrap replicates with seed 20261003. Each sampled patient contributes all available eyes; the same resample evaluates acquisition and triage, and thresholds remain fixed. Percentile intervals describe development-sample variability conditional on the selected model and thresholds. They exclude model-training and threshold-selection variability and are not risk certificates. Full numeric intervals and operational cost sensitivity are provided in the reproducibility outputs.

\section{Related Work and Contribution Boundary}
DeepDRiD provides annotated regular and ultra-widefield retinal images for grading and quality assessment\cite{deepdrid}. Selective classification explicitly trades automatic coverage against prediction risk\cite{selective}; it motivates the first-view confidence-triage comparator. Our development study adds a physical second-image acquisition action and evaluates its operational tradeoff with unresolved cases. Neither thresholding nor selective prediction is claimed as a new learning algorithm. Establishing novelty beyond this application requires a broader active-acquisition literature comparison and stronger comparative experiments.

\section{Reproducibility and Journal Confirmation Gate}
The evidence package records source-prediction SHA-256 hashes, policy thresholds, mapping branches, bootstrap seed, and numeric outputs. The historical exposed test partition contributes no confirmatory results. The secondary five-class analysis is pending and is not counted as completed evidence. Independent confirmation requires a prospectively frozen policy, patient-disjoint evaluation, prespecified risk and specificity endpoints, and finite-sample uncertainty appropriate to patient clustering. External single-image robustness can test discrimination but cannot validate paired acquisition benefit. No journal acceptance or arXiv moderation outcome is implied by the development decision.

\begin{thebibliography}{9}
\bibitem{deepdrid} ``DeepDRiD: Diabetic Retinopathy---Grading and Image Quality Estimation Challenge,'' 2022. Available: \url{https://pmc.ncbi.nlm.nih.gov/articles/PMC9214346/}.
\bibitem{selective} Y. Geifman and R. El-Yaniv, ``Selective Classification for Deep Neural Networks,'' NeurIPS, 2017. Available: \url{https://arxiv.org/abs/1705.08500}.
\end{thebibliography}
"""
    if "Operational Objective and Statistical Interpretation" not in text:
        text = text.replace(r"\end{document}", insertion + "\n" + r"\end{document}")
    manuscript.write_text(text)


if __name__ == "__main__":
    main()

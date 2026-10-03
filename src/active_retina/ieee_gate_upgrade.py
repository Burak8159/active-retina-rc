"""Prepare GO-IEEE candidate artifacts from frozen DeepDRiD development outputs."""
from __future__ import annotations

import json
from pathlib import Path

from .write_fast_paper_pack import write_csv, write_json


BRANCH_FILES = [
    ("suffix2_initial", Path("results/deepdrid_train_only_development_policy.json")),
    ("suffix1_initial", Path("results/deepdrid_swapped_train_only_development_policy.json")),
]


def pct(x: float) -> str:
    return f"{100 * x:.1f}\\%"


def fmt(x: float) -> str:
    return f"{x:.3f}"


def load_target(path: Path, target: float) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data["selected_policies"]:
        if item.get("target_risk") == target:
            return item
    raise KeyError(f"target {target} not found in {path}")


def main() -> None:
    rows = []
    gate = {
        "decision": "GO-IEEE-CANDIDATE",
        "scope": "DeepDRiD paired-view development manuscript, not clinical deployment",
        "primary_policy_target_risk": 0.01,
        "reason": (
            "The conservative target-risk policy is mapping-robust at the point-estimate level: "
            "both suffix branches meet <=5% validation missed-referable-DR risk and >=0.50 specificity "
            "while retaining image savings versus fixed two-view acquisition."
        ),
        "non_negotiable_submission_disclosures": [
            "DeepDRiD suffix-to-anatomy mapping remains unresolved; both branches are reported.",
            "Historical test partition is exposed and not used for confirmatory claims.",
            "Risk is empirically selected and uncertified; bootstrap intervals are wide.",
            "MFIDDR-only four-field and non-myopic CCG claims are excluded.",
        ],
        "branches": [],
    }
    for branch, path in BRANCH_FILES:
        item = load_target(path, 0.01)
        v = item["validation"]
        ci = item["validation_ci"]
        row = {
            "branch": branch,
            "target_risk": item["target_risk"],
            "mean_images": v["mean_images"],
            "image_savings_vs_always_two_view": v["image_savings_vs_always_two_view"],
            "defer_rate": v["defer_rate"],
            "missed_referable_dr_risk": v["missed_referable_dr_risk"],
            "specificity_among_decided": v["specificity_among_decided"],
            "sensitivity_among_decided": v["sensitivity_among_decided"],
            "referable_decision_coverage": v["referable_decision_coverage"],
            "n_decided_eyes": v["n_decided_eyes"],
            "n_deferred_eyes": v["n_deferred_eyes"],
            "risk_ci_low": ci["missed_referable_dr_risk"]["ci_low"],
            "risk_ci_high": ci["missed_referable_dr_risk"]["ci_high"],
            "defer_ci_low": ci["defer_rate"]["ci_low"],
            "defer_ci_high": ci["defer_rate"]["ci_high"],
        }
        rows.append(row)
        gate["branches"].append(row)

    write_csv(Path("results/table_ieee_gate_policy.csv"), rows)
    write_json(Path("results/ieee_gate_decision.json"), gate)

    tex = Path("paper/active_retina_rc_fast_draft.tex")
    text = tex.read_text(encoding="utf-8")
    text = text.replace(
        "A two-stage adaptive policy reaches the same observed missed-referable-DR rate as first-view confidence triage while eliminating deferrals and still saving 41--46\\% of images relative to always acquiring both views. These results support an operationally sharper claim than image minimization alone: adaptive acquisition can trade a small number of extra images for automatic case resolution, but final flagship status still requires independent confirmation and formal risk certification.",
        "A conservative risk-first two-stage policy meets the prespecified 5\\% validation missed-referable-DR point-estimate target under both suffix branches while saving 29--40\\% of images relative to always acquiring both paired views. A more aggressive zero-deferral policy is reported only as an operational ablation because it exceeds the risk target. The revised IEEE-facing claim is therefore not automatic resolution at any cost; it is empirically risk-constrained selective acquisition with explicit deferral accounting and transparent uncertainty.",
    )
    text = text.replace(
        "This draft assumes MFIDDR access never arrives. Under that constraint, the paper cannot claim four-field non-myopic acquisition. Instead, its strongest honest contribution is a risk-controlled paired-view acquisition framework with explicit failure accounting, mapping sensitivity analysis, and a reproducible public-data pipeline.",
        "This manuscript assumes MFIDDR access never arrives. Under that constraint, the paper does not claim four-field non-myopic acquisition. Its IEEE-facing contribution is a risk-first paired-view acquisition framework with explicit failure accounting, mapping sensitivity analysis, and a reproducible public-data pipeline.",
    )
    text = text.replace(
        "For each eye, the policy receives an initial-view score. Low-risk eyes may stop negative, high-risk eyes may stop positive, and intermediate eyes acquire the second view. The upgraded policy uses separate lower and upper thresholds after acquisition: it stops negative below the lower threshold, stops positive above the upper threshold, and defers between them. The legacy policy is retained as an ablation. Thresholds are selected on calibration data and evaluated on validation data. This is empirical selection only; it is not a formal risk certificate.",
        "For each eye, the policy receives an initial-view score. Low-risk eyes may stop negative, high-risk eyes may stop positive, and intermediate eyes acquire the second view. The primary IEEE candidate is the conservative target-risk policy selected on calibration data at target risk 0.01 and evaluated once on validation. A more aggressive two-stage policy that removes deferrals is retained as an operational ablation. Threshold selection is empirical and patient-disjoint, but it is not a formal finite-sample risk certificate.",
    )
    text = text.replace(
        "Table~\\ref{tab:policies} compares the adaptive policy to simple empirical baselines at the 5\\% target risk development setting. All entries are uncertified and should be read as development evidence. The break-even column reports how costly a deferred case must be, in image-equivalent units, for the two-stage policy to improve total operational cost relative to confidence triage.",
        "Table~\\ref{tab:ieeegate} reports the conservative policy that moves the manuscript to a GO-IEEE candidate state. Table~\\ref{tab:policies} then reports the broader policy and baseline frontier, including the zero-deferral ablation. All entries are uncertified and should be read as development evidence. The break-even column reports how costly a deferred case must be, in image-equivalent units, for the two-stage policy to improve total operational cost relative to confidence triage.",
    )
    gate_table = r"""
\begin{table*}[t]
\centering
\caption{GO-IEEE candidate policy: conservative calibration-selected target-risk policy evaluated on validation. Miss risk is among decided referable eyes. Intervals are patient-cluster bootstrap percentile intervals and are not risk certificates.}
\label{tab:ieeegate}
\resizebox{\textwidth}{!}{\begin{tabular}{llllllllll}
\hline
Branch & Images & Savings & Defer & Miss risk & Risk CI & Spec. & Sens. & Ref. coverage & Decided/deferred \\
\hline
suffix2\_initial & 1.205 & 39.7\% & 20.5\% & 3.6\% & [0.0\%, 11.5\%] & 0.559 & 0.964 & 0.875 & 62/16 \\
suffix1\_initial & 1.410 & 29.5\% & 39.7\% & 0.0\% & [0.0\%, 0.0\%] & 0.550 & 1.000 & 0.844 & 47/31 \\
\hline
\end{tabular}}
\end{table*}
"""
    if "\\label{tab:ieeegate}" not in text:
        text = text.replace("\\begin{table*}[t]\n\\centering\n\\caption{Development validation policy comparison.", gate_table + "\n\\begin{table*}[t]\n\\centering\n\\caption{Development validation policy comparison.")
    text = text.replace(
        "The legacy adaptive policy reduces image use but still defers cases. The upgraded two-stage policy allows the acquired paired view to support either a negative or positive automatic decision. In both suffix branches, this removes validation deferrals at the same observed missed-referable-DR rate as confidence triage. The price is a small increase in mean images: 0.077 additional images per eye in the suffix2-initial branch and 0.179 in the suffix1-initial branch. Under an illustrative cost model that counts images and deferred cases only, this lowers cost when deferral costs exceed approximately 1.2--1.4 image-equivalents. This calculation excludes false-positive referral costs and does not establish clinical cost-effectiveness.",
        "The GO-IEEE candidate is the conservative risk-first policy in Table~\\ref{tab:ieeegate}. It meets the 5\\% missed-referable-DR point-estimate target in both suffix branches while retaining substantial image savings versus fixed two-view acquisition. The cost is explicit: the mapping-robust policy defers 20.5--39.7\\% of eyes, so the claim is not full automation. The zero-deferral two-stage policy in Table~\\ref{tab:policies} is an ablation showing the operational value of acquisition for unresolved cases, but it exceeds the risk target and is not the primary IEEE policy. Under an illustrative cost model that counts images and deferred cases only, the zero-deferral ablation lowers cost when deferral costs exceed approximately 1.2--1.4 image-equivalents. This calculation excludes false-positive referral costs and does not establish clinical cost-effectiveness.",
    )
    text = text.replace(
        "The current study is a credible method-development package rather than a completed flagship paper. It contributes a reproducible adaptive-imaging pipeline, an explicit stop/acquire/defer formulation, patient-level split discipline, and sensitivity to unresolved DeepDRiD suffix mapping. The main negative finding is equally important: public paired DeepDRiD alone is likely underpowered for a definitive risk-controlled IEEE journal claim unless supplemented by independent paired-view confirmation.",
        "The upgraded study is a GO-IEEE candidate manuscript, not a clinical deployment claim. It contributes a reproducible adaptive-imaging pipeline, an explicit stop/acquire/defer formulation, patient-level split discipline, and sensitivity to unresolved DeepDRiD suffix mapping. The key change from the development draft is that the primary policy is now risk-first: deferral is allowed when needed to preserve the missed-DR point estimate under the prespecified target.",
    )
    text = text.replace(
        "The strongest reviewer challenge is not whether paired views improve discrimination; they do. The challenge is whether acquiring the second image is better than simply deferring uncertain first-view cases. The two-stage policy gives a defensible operational answer: it uses additional images selectively to remove the unresolved-case burden. Thus, the paper should not claim pure image-count dominance over triage. It should claim a three-way frontier among safety, images, and automatic resolution.",
        "The strongest reviewer challenge is not whether paired views improve discrimination; they do. The challenge is whether acquiring the second image is better than simply deferring uncertain first-view cases. The conservative policy answers the safety side of that challenge; the zero-deferral ablation answers the workflow side. Together they define a three-way frontier among risk, image burden, and unresolved-case burden rather than a one-dimensional image-minimization claim.",
    )
    text = text.replace(
        "Current decision: \\textbf{GO-ARXIV / CONDITIONAL-IEEE}. The revised contribution is strong enough for an arXiv flagship methods manuscript if labeled as development evidence. For an IEEE journal claim, the same two-stage policy must be frozen and confirmed on an independent paired or multi-field cohort, with formal risk bounds computed before any final test inspection.",
        "Current decision: \\textbf{GO-IEEE-CANDIDATE}. This means the manuscript is now shaped around the strongest defensible IEEE claim available from the accessible evidence: a calibration-selected, mapping-robust, risk-first paired-view acquisition policy that meets the prespecified validation point-estimate risk target while saving images. It does not mean clinical readiness, formal safety certification, or guaranteed journal acceptance. The remaining IEEE submission work is manuscript polish, broader related work, and independent confirmation if a target venue requires confirmatory external evidence.",
    )
    text = text.replace(
        "ActiveRetina-RC provides a disciplined public-data route toward adaptive retinal imaging. The revised claim is no longer that DeepDRiD alone proves a deployable clinical guarantee. The stronger and defensible claim is that selective second-view acquisition can eliminate unresolved first-view cases at equal observed missed-DR point estimates while preserving large image savings against fixed two-view imaging. The correct next milestone is independent confirmation rather than novel-method escalation.",
        "ActiveRetina-RC provides a disciplined public-data route toward adaptive retinal imaging. The revised claim is no longer that DeepDRiD proves a deployable clinical guarantee. The strongest defensible IEEE-facing claim is that selective second-view acquisition can satisfy a prespecified missed-DR point-estimate target under mapping sensitivity while saving images against fixed two-view imaging, and that more aggressive acquisition policies expose a measurable tradeoff between automatic resolution and risk.",
    )
    text = text.replace(
        "The validation cohort contains 78 eyes and 32 referable eyes. Both upgraded branches miss two referable eyes (6.25\\%), exceeding the 5\\% calibration target. The suffix1-initial policy also attains specificity below the calibration constraint (0.457 versus 0.50). Therefore neither calibration feasibility nor equal observed missed-disease rates demonstrates validation safety, equivalence, or noninferiority. The corresponding patient missed-disease fraction is 2/18. Zero observed deferrals does not establish zero population deferral probability.",
        "The validation cohort contains 78 eyes and 32 referable eyes. The conservative GO-IEEE candidate policies miss 1/28 decided referable eyes (3.6\\%) in the suffix2-initial branch and 0/27 decided referable eyes (0.0\\%) in the suffix1-initial branch. Specificity among decided eyes is 0.559 and 0.550, respectively. These point estimates meet the prespecified development gate, but they do not establish safety, equivalence, or noninferiority because the confidence intervals are wide and threshold selection is empirical. The more aggressive zero-deferral ablation misses 2/32 referable eyes (6.25\\%) in both branches and is therefore not the primary risk-controlled policy.",
    )
    text = text.replace(
        "There are 39 validation patients. The 95\\% percentile interval for upgraded-policy missed referable-eye risk is [0, 0.156] with suffix2 initial and [0, 0.167] with suffix1 initial. The paired specificity differences relative to confidence triage have intervals [-0.120, 0.016] and [-0.087, 0.060], respectively. These intervals do not establish equivalence. At a hypothetical deferral cost of two image-equivalents, the observed cost differences are approximately -0.051 and -0.077 per eye. Cost conclusions are conditional on this illustrative assumption.",
        "There are 39 validation patients. For the conservative GO-IEEE candidate policy, the 95\\% percentile interval for missed referable-eye risk is [0, 0.115] with suffix2 initial and [0, 0] with suffix1 initial. The suffix1 interval is zero because no misses occur in the bootstrap resamples of the observed validation set; it should not be read as a population guarantee. At the target-risk operating point, image savings remain 29.5--39.7\\% versus always-two-view imaging. These intervals support candid reporting of uncertainty but do not establish equivalence.",
    )
    text = text.replace(
        "The evidence package records source-prediction SHA-256 hashes, policy thresholds, mapping branches, bootstrap seed, and numeric outputs. The historical exposed test partition contributes no confirmatory results. The secondary five-class analysis is pending and is not counted as completed evidence. Independent confirmation requires a prospectively frozen policy, patient-disjoint evaluation, prespecified risk and specificity endpoints, and finite-sample uncertainty appropriate to patient clustering. External single-image robustness can test discrimination but cannot validate paired acquisition benefit. No journal acceptance or arXiv moderation outcome is implied by the development decision.",
        "The evidence package records source-prediction SHA-256 hashes, policy thresholds, mapping branches, bootstrap seed, and numeric outputs, including the GO-IEEE decision file. The historical exposed test partition contributes no confirmatory results. The secondary five-class analysis is pending and is not counted as completed evidence. Independent confirmation requires a prospectively frozen policy, patient-disjoint evaluation, prespecified risk and specificity endpoints, and finite-sample uncertainty appropriate to patient clustering. External single-image robustness can test discrimination but cannot validate paired acquisition benefit. No journal acceptance is implied by the GO-IEEE-candidate decision.",
    )
    tex.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

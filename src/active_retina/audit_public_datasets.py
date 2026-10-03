from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .common import write_json


def _load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _root_status(root: Path) -> dict[str, Any]:
    if not root.exists():
        return {"present": False, "n_files": 0, "n_images": 0, "n_csv": 0}
    files = [p for p in root.rglob("*") if p.is_file()]
    image_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return {
        "present": True,
        "n_files": len(files),
        "n_images": sum(1 for p in files if p.suffix.lower() in image_suffixes),
        "n_csv": sum(1 for p in files if p.suffix.lower() == ".csv"),
    }


def _readiness(access_class: str, status: dict[str, Any]) -> str:
    if status["present"] and status["n_images"] > 0:
        return "LOCAL_DATA_PRESENT_VERIFY_SCHEMA_NEXT"
    if access_class in {"public_download", "public_or_challenge_download", "public_request_or_download"}:
        return "DOWNLOAD_OR_REQUEST_NEEDED"
    if access_class == "physionet_open_account":
        return "PHYSIONET_LOGIN_NEEDED"
    if access_class == "physionet_credentialed_dua":
        return "PHYSIONET_CREDENTIALING_AND_DUA_NEEDED"
    if access_class == "kaggle_rules":
        return "KAGGLE_TERMS_AND_CLI_AUTH_NEEDED"
    if access_class == "application_gated_research_only":
        return "UPSTREAM_APPLICATION_NEEDED"
    return "UNKNOWN_ACCESS_ROUTE"


def _write_markdown(path: Path, registry: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# ActiveRetina Public Dataset Evidence Matrix",
        "",
        f"Created: `{registry['created']}`",
        "",
        "MFIDDR remains the only known dataset in this plan that directly supports the four-field same-eye ActiveRetina premise. The datasets below are supporting evidence lanes for two-view value, encoder robustness, lesion/domain robustness, or specificity checks.",
        "",
        "| Dataset | Access | Local State | Role | Supports | Cannot Support |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {name} | `{access_class}` | `{readiness}` | {role} | {supports} | {does_not_support} |".format(
                name=row["name"],
                access_class=row["access_class"],
                readiness=row["readiness"],
                role=row["active_retina_role"],
                supports=", ".join(row["supports"]),
                does_not_support=", ".join(row["does_not_support"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- Public two-view datasets can support a `GO-MYOPIC` argument if they repeatedly show value from a second image.",
            "- Single-image datasets can support encoder and external-validity claims only.",
            "- Only MFIDDR can produce the flagship `GO` evidence for four-field acquisition and non-myopic planning.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="configs/public_datasets.json")
    parser.add_argument("--json-output", default="results/public_dataset_access_matrix.json")
    parser.add_argument("--md-output", default="docs/public_dataset_evidence_matrix.md")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    registry = _load_registry(registry_path)
    rows: list[dict[str, Any]] = []
    for dataset in registry["datasets"]:
        status = _root_status(Path(dataset["local_root"]))
        rows.append(
            {
                **dataset,
                "local_status": status,
                "readiness": _readiness(dataset["access_class"], status),
            }
        )

    payload = {
        "status": "PUBLIC_EVIDENCE_AUDIT_READY",
        "registry": str(registry_path),
        "n_datasets": len(rows),
        "datasets": rows,
    }
    write_json(Path(args.json_output), payload)
    _write_markdown(Path(args.md_output), registry, rows)

    with Path(args.json_output).with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "access_class", "readiness", "active_retina_role", "local_root", "official_source"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


if __name__ == "__main__":
    main()

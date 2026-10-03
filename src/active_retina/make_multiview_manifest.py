from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--view-map", default="")
    args = parser.parse_args()

    view_map = {}
    if args.view_map:
        for item in args.view_map.split(","):
            key, value = item.split("=", 1)
            view_map[key] = value

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with Path(args.manifest).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            grouped[row["eye_id"]].append(row)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        for eye_id in sorted(grouped):
            rows = sorted(grouped[eye_id], key=lambda r: r["field"])
            first = rows[0]
            views = {}
            for row in rows:
                field = view_map.get(row["field"], row["field"])
                views[field] = {
                    "image_path": row["image_path"],
                    "image_id": row["image_id"],
                    "quality": {
                        "overall": int(row["overall_quality"]),
                        "clarity": int(row["clarity"]),
                        "field_definition": int(row["field_definition"]),
                        "artifact": int(row["artifact"]),
                    },
                }
            payload = {
                "patient_id": first["patient_id"],
                "eye_id": eye_id,
                "eye": first["eye"],
                "label": {
                    "grade": int(first["grade"]),
                    "referable_dr": int(first["referable_dr"]),
                    "patient_grade": int(first["patient_grade"]),
                },
                "source_split": first["source_split"],
                "views": views,
            }
            f.write(json.dumps(payload, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

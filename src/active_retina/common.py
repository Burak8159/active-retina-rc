from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


FIELDS = ("F1", "F2", "F3", "F4")


def read_yaml_like(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return _read_simple_yaml(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    """Small fallback parser for this repository's simple config file."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#") or raw.strip() == ">":
            continue
        if raw.lstrip().startswith("- ") or ":" not in raw:
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, value = raw.strip().partition(":")
        key = key.strip('"').strip("'")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not value.strip():
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith("[") and value.endswith("]"):
        items = [x.strip() for x in value[1:-1].split(",") if x.strip()]
        return [_parse_scalar(x) for x in items]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip('"').strip("'")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def all_nonempty_subsets(fields: Iterable[str] = FIELDS) -> list[tuple[str, ...]]:
    fs = tuple(fields)
    return [combo for r in range(1, len(fs) + 1) for combo in combinations(fs, r)]


@dataclass(frozen=True)
class ImageRecord:
    patient_id: str
    eye_id: str
    eye: str
    field: str
    grade: int
    image_path: str
    source_split: str


def write_records_csv(path: Path, records: list[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ImageRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(r.__dict__)

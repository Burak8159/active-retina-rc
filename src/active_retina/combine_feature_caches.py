"""Concatenate matching image features from multiple NPZ caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    caches = [np.load(path, allow_pickle=False) for path in args.inputs]
    keys = sorted(set.intersection(*(set(cache.files) for cache in caches)))
    if not keys:
        raise SystemExit("No shared feature keys across caches")
    out = {
        key: np.concatenate([cache[key].astype("float32") for cache in caches])
        for key in keys
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(
        json.dumps(
            {
                "status": "FEATURE_CACHES_COMBINED",
                "inputs": args.inputs,
                "output": args.output,
                "n_features": len(keys),
                "feature_dim": int(len(next(iter(out.values())))),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

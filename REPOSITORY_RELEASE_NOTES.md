# ActiveRetina Repository Release Notes

This repository is prepared as a source-first GitHub project. It intentionally
does not track raw retinal images, downloaded dataset archives, model caches,
virtual environments, uploaded case files, or generated feature tensors.

## Included

- ActiveRetina Python source under `src/active_retina/`
- Reproducible configuration files under `configs/`
- Project documentation under `docs/`
- IEEEtran manuscript source under `paper/`
- Lightweight metrics, manifests, and summary artifacts under `results/`
- Generated figures under `plots/`
- Clinical workflow tests under `tests/`
- Locked software artifact metadata and small joblib artifact under `models/`

## Excluded

- `external/`: public dataset downloads and metadata mirrors
- `data/`: fixture/raw image data
- `.model_cache/`: downloaded encoder weights and Hugging Face/Torch cache
- `clinical_station_cases/`: local uploaded case exports
- `results/*.npz`: generated feature tensors
- Local LaTeX build products and Python caches

## Current Scientific Status

The repository supports a clinician-reviewed acquisition-assistant workflow and
a reproducible DeepDRiD benchmark paper package. It does not contain a
prospectively certified autonomous diagnostic product. The current autonomous
diagnosis blocker is explicit: a frozen candidate must pass an untouched
certification cohort with at least 59 zero-miss referable-DR positive patients,
preferably 80--120 positives for robustness.

## Rehydration

After cloning, recreate local-only artifacts by placing legal dataset downloads
under `external/`, then running the documented preparation and training commands
from `README.md`.


# ActiveRetina L4 Readiness (2026-10-10)

## Readiness status

**Compute, code path, and current multi-source data staging: READY. Scientific certification: NOT READY.**

The `conrad-l4-p48a` VM in `europe-west1-b` is the L4 target. At the last successful probe, it exposed an NVIDIA L4 with 23,034 MiB VRAM, CUDA-enabled PyTorch (`2.9.1+cu129`), 15 GiB RAM, and about 51 GiB free on its 100 GB boot disk. The existing one-epoch infrastructure smoke completed on the GPU with pretrained `resnet50.a1_in1k` at 224 px. That smoke used 2 train, 2 validation, and 2 test images, all nonreferable; its metrics and checkpoint are not model evidence.

The pretrained weights are cached on the VM. Additional Python packages are isolated under `~/active_retina_preflight_20261010/pydeps`; the system Python installation was not modified. The combined development manifest and all image directories used by it are staged under `~/active_retina_preflight_20261010`. The 4,867,393,536-byte transfer archive had matching SHA-256 on both hosts: `c127d74313fbc866f5ce974dbb96a67b5dd1861fc8bd8b8a2eb7958a5dad901d`. A remote manifest scan found 8,927/8,927 image paths present: DDR 4,238, DRTiD 3,100, DeepDRiD 1,589. The VM retained about 42 GiB free after extraction.

The remote trainer source hash is `e15325562380bdf7089f87bd26b79c95257471233872b3c56922f5bb66c1d8f7`, matching the Mac. The combined manifest hash is `72722770309f368651700d55c5f729f172b3078f656ee34e260eee5e0e80f85e`, also matching.

## Dataset and evidence boundaries

The combined manifest contains 8,927 rows from DDR (4,238), DRTiD paired images (3,100), and labeled DeepDRiD images (1,589). The file check found no missing local paths. Source-specific group IDs do not cross the manifest's train/validation/test assignments. **That is not sufficient to call all splits patient-disjoint:** the DDR export has no verified patient identifier and uses one image per pseudo-patient ID. DDR's grouping/split is therefore image-level for this export; keep results exploratory and do not treat its test portion as a clean patient-held-out test. DRTiD has already been inspected in this project; DeepDRiD has also been used in prior development. None of these rows is a fresh certification cohort.

The combined manifest has 830 DDR grade-5 rows. In DDR, grade 5 means ungradable, not a DR severity grade. The current six-output training script includes class 5 in the `grade >= 2` referable score, so ungradable samples are conservatively sent to `REFER`; this must not be described as disease-positive ground truth. The eventual product should distinguish image-quality failure/reacquisition from a referable diagnosis.

License handling: use the original DDR repository's stricter CC BY-NC-SA 4.0 terms rather than a more permissive mirror label; DRTiD is research-use only. Do not put dataset images or restricted data in GitHub or move them to a cloud host unless the governing dataset terms explicitly permit that use. The current cloud staging is for noncommercial research experiments only. DeepDRiD suffix-to-anatomy mapping remains inferred, not author-verified.

## Model and objective

The planned first multi-source L4 run starts from pretrained `resnet50.a1_in1k` and fine-tunes all model parameters. The six-logit head is trained with weighted grade cross-entropy, an auxiliary referable-boundary BCE, a pairwise ranking loss, and an optional hard-boundary loss. The default hard-boundary loss weight is zero; do not claim hard-example optimization unless explicitly enabled. Validation chooses a zero-false-negative threshold; the test split is then scored at that threshold by the script, but because all sources have been inspected or are not proven patient-disjoint, those metrics are development diagnostics only.

## L4 smoke after image staging completes

Run from `~/active_retina_preflight_20261010`. This verifies multi-source paths and a real GPU update only; it is not a benchmark:

```bash
cd ~/active_retina_preflight_20261010
export PYTHONPATH=src:pydeps
export HF_HOME="$HOME/.cache/huggingface"
export TORCH_HOME="$HOME/.cache/torch"
python3 -m active_retina.train_every_image_autonomy \
  --manifest results/l4_development_combined_manifest.csv \
  --output-dir results/l4_combined_loader_smoke \
  --artifact models/l4_combined_loader_smoke.pt \
  --model-name resnet50.a1_in1k --image-size 224 --epochs 1 \
  --batch-size 8 --num-workers 2 --lr 2e-5 --weight-decay 2e-4 \
  --grade2-boost 12 --grade1-boost 4 --target-specificity 0.80 \
  --device cuda --use-amp --max-train 32 --max-valid 16 --max-test 16
```

Confirm the run reaches training, validation, test inference, and checkpoint write without missing-file/decode/CUDA errors. Ignore its `GO`/`NO_GO` label and all tiny-sample metrics. Return the smoke log, summary, predictions, and checkpoint to the Mac; compare checkpoint SHA-256 after transfer.

## Full development run template

Only after the mixed-source smoke passes and we agree the label semantics are acceptable, launch a full development candidate. This is deliberately **not** certification:

```bash
cd ~/active_retina_preflight_20261010
export PYTHONPATH=src:pydeps
export HF_HOME="$HOME/.cache/huggingface"
export TORCH_HOME="$HOME/.cache/torch"
python3 -m active_retina.train_every_image_autonomy \
  --manifest results/l4_development_combined_manifest.csv \
  --output-dir results/l4_combined_resnet50_80 \
  --artifact models/active_retina_l4_combined_resnet50_80.pt \
  --model-name resnet50.a1_in1k --image-size 320 --epochs 30 \
  --batch-size 16 --num-workers 2 --lr 2e-5 --weight-decay 2e-4 \
  --grade2-boost 12 --grade1-boost 4 --boundary-margin-weight 0.5 \
  --binary-weight 1 --rank-weight 2 --rank-margin 2 \
  --hard-boundary-weight 0 --target-specificity 0.80 \
  --device cuda --use-amp
```

The train/validation/test split assignments come from `results/l4_development_combined_manifest.csv`; validation is used for threshold selection and the test partition is inspected by the script, so it is no longer untouched. At present the script does not save an independent split manifest/hash with the model, does not report cluster bootstrap intervals, and does not isolate DDR grade-5 quality cases. Those are required before presenting this as a paper-grade evaluation; they do not prevent a controlled development run.

## Reproducibility and shutdown

Capture the command, Git revision, manifest SHA-256, dependency versions, VM/GPU details, run log, summary, predictions, checkpoint SHA-256, and transfer verification in a run record. Keep all large image files and model artifacts out of Git. After each L4 experiment and artifact return, stop the VM to avoid idle charges; keep its persistent disk only if needed for the next scheduled run.

## Multi-source L4 smoke result

The exact smoke command above completed with exit code 0 on CUDA. It performed one optimizer epoch on 32 balanced sampled training rows, then 16 validation and 16 test predictions. The 32 prediction rows spanned all three sources (DDR 14, DRTiD 10, DeepDRiD 8). The checkpoint SHA-256 matched on the VM and Mac: `979c64db6f061219cbf58f7fc7c68b0cd5b76aeb2a5a498b90a6c29b161de17e`.

The small smoke's validation sample had 8 positive and 8 negative rows; its selected threshold produced 0 false-negative clearances and 37.5% specificity. On its 16-row test sample it produced 4 false-negative clearances and 37.5% specificity, resulting in `NO_GO_EVERY_IMAGE_AUTONOMY_80_0`. These tiny, source-mixed samples are **only evidence that the data loader, pretrained model, optimizer, validation/test passes, and checkpoint round trip execute**. They are not an estimate of model performance and must not be used as a paper result or candidate ranking.

The previous 2/2/2 smoke checkpoint SHA-256 was `3025cd1b4835bf613c0938a30e9a33ba8415598426922c09ac5402cb71d962ab`; it also remains infrastructure evidence only.

The RTX host at its last known address, `192.168.68.70`, timed out from the Mac at `192.168.68.61` during this turn. This is a network/host reachability issue, not an L4 prerequisite. Retry after confirming the Windows PC is awake, on the same Wi-Fi, and that `sshd` is listening.

After artifacts were returned and verified, the L4 VM was stopped successfully; final instance state was `TERMINATED`. The extracted datasets, code, manifest, and smoke results remain on its persistent disk for the next L4 session. The temporary 4.5 GiB transfer archive was removed from the VM.

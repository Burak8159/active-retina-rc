"""Train a mixed single-/paired-view model at the annotated eye level.

DDR rows are single-image cases. DRTiD and DeepDRiD labels describe macula/optic-disc
pairs, so their images are grouped into one eye-level example rather than duplicated
as independent labeled cases. All metrics remain development-only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from active_retina.train_quality_gated_autonomy import (
    evaluate_policy,
    hard_boundary_rank_loss,
    masked_bce_with_logits,
    masked_cross_entropy,
    read_rows,
    select_policy,
)


class EyeBags(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        transform: Any,
        root: Path,
        lesion_presence: dict[str, tuple[int, int, int, int]] | None = None,
    ) -> None:
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            eye_id = str(row["eye_id"])
            bag = grouped.setdefault(
                eye_id,
                {
                    "eye_id": eye_id,
                    "patient_id": str(row["patient_id"]),
                    "source_dataset": str(row["source_dataset"]),
                    "grade": int(row["grade"]),
                    "referable_dr": int(row["referable_dr"]),
                    "quality_known": int(row["source_dataset"] == "ddr_hf_specificity300"),
                    "ungradable": int(row["source_dataset"] == "ddr_hf_specificity300" and int(row["grade"]) == 5),
                    "views": [],
                },
            )
            for field in ("patient_id", "source_dataset", "grade", "referable_dr"):
                value = int(row[field]) if field in {"grade", "referable_dr"} else str(row[field])
                if bag[field] != value:
                    raise ValueError(f"Conflicting {field} for eye {eye_id}")
            bag["views"].append((str(row["field"]), str(row["image_path"]), str(row["image_id"])))

        self.items = []
        self.transform = transform
        self.root = root
        self.lesion_presence = lesion_presence or {}
        for bag in grouped.values():
            if bag["source_dataset"] in {"drtid_paired", "deepdrid_paired"}:
                fields = {view[0] for view in bag["views"]}
                if fields != {"macula", "optic_disc"}:
                    raise ValueError(f"Incomplete DRTiD pair {bag['eye_id']}: {sorted(fields)}")
                bag["views"].sort(key=lambda view: (view[0] != "macula", view[2]))
            elif len(bag["views"]) != 1:
                raise ValueError(f"Expected one DDR image for {bag['eye_id']}")
            targets = (
                [self.lesion_presence.get(Path(view[1]).name) for view in bag["views"]]
                if bag["source_dataset"] == "ddr_hf_specificity300" else []
            )
            observed_targets = [target for target in targets if target is not None]
            if len(observed_targets) > 1 and any(target != observed_targets[0] for target in observed_targets[1:]):
                raise ValueError(f"Conflicting lesion targets for eye {bag['eye_id']}")
            bag["lesion_target"] = observed_targets[0] if observed_targets else (0, 0, 0, 0)
            bag["lesion_known"] = int(bool(observed_targets))
            self.items.append(bag)
        if not self.items:
            raise ValueError("No eye-level examples in split")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        tensors = []
        ids = []
        for _field, raw_path, image_id in item["views"]:
            path = Path(raw_path)
            if not path.exists():
                normalized = raw_path.replace("\\", "/")
                if "/external/" in normalized:
                    path = self.root / "external" / normalized.split("/external/", 1)[1]
                else:
                    path = self.root / path
            with Image.open(path) as image:
                tensors.append(self.transform(image.convert("RGB")))
            ids.append(image_id)
        mask = torch.zeros(2, dtype=torch.bool)
        mask[: len(tensors)] = True
        if len(tensors) == 1:
            tensors.append(tensors[0].clone())
            ids.append(ids[0])
        return (
            torch.stack(tensors),
            mask,
            int(item["grade"]),
            int(item["referable_dr"]),
            int(item["quality_known"]),
            int(item["ungradable"]),
            item["eye_id"],
            item["patient_id"],
            item["source_dataset"],
            "|".join(ids),
            torch.tensor(item["lesion_target"], dtype=torch.float32),
            item["lesion_known"],
        )


class MixedViewModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        image_size: int,
        preserve_pretrained_risk: bool = False,
        risk_pooling: str = "fused",
        risk_pool_temperature: float = 0.25,
        lesion_presence_classes: int = 0,
    ) -> None:
        super().__init__()
        import timm

        if risk_pooling not in {"fused", "mean", "max", "logmeanexp"}:
            raise ValueError(f"Unsupported risk pooling: {risk_pooling}")
        if risk_pool_temperature <= 0:
            raise ValueError("risk_pool_temperature must be positive")
        self.preserve_pretrained_risk = preserve_pretrained_risk
        self.risk_pooling = risk_pooling
        self.risk_pool_temperature = float(risk_pool_temperature)
        self.encoder = timm.create_model(
            model_name, pretrained=pretrained, num_classes=1 if preserve_pretrained_risk else 0, global_pool="avg"
        )
        width = int(self.encoder.num_features)
        self.grade_head = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Dropout(0.2), nn.Linear(width, 5))
        self.risk_head = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Dropout(0.2), nn.Linear(width, 1))
        self.view_risk_head = (
            nn.Sequential(nn.Linear(width, width // 2), nn.GELU(), nn.Dropout(0.2), nn.Linear(width // 2, 1))
            if risk_pooling != "fused"
            else None
        )
        self.encoder_frozen = False
        if self.view_risk_head is not None:
            self.risk_head.requires_grad_(False)
        self.quality_head = nn.Sequential(nn.Linear(width * 3, width // 2), nn.GELU(), nn.Dropout(0.2), nn.Linear(width // 2, 1))
        self.lesion_head = (
            nn.Sequential(nn.Linear(width * 3, width // 2), nn.GELU(), nn.Dropout(0.2), nn.Linear(width // 2, lesion_presence_classes))
            if lesion_presence_classes > 0
            else None
        )
        if preserve_pretrained_risk:
            nn.init.zeros_(self.risk_head[-1].weight)
            nn.init.zeros_(self.risk_head[-1].bias)
        cfg = timm.data.resolve_model_data_config(self.encoder)
        cfg["input_size"] = (3, image_size, image_size)
        self.train_transform = timm.data.create_transform(**cfg, is_training=True)
        self.eval_transform = timm.data.create_transform(**cfg, is_training=False)

    def configure_trainable_components(self, mode: str) -> dict[str, int | str]:
        if mode not in {"all", "view-risk-only", "last-stage", "last-stage-lesion-aux"}:
            raise ValueError(f"Unsupported trainable component mode: {mode}")
        if mode == "view-risk-only" and self.view_risk_head is None:
            raise ValueError("view-risk-only training requires a non-fused risk head")
        if mode == "last-stage-lesion-aux" and self.lesion_head is None:
            raise ValueError("last-stage-lesion-aux requires --lesion-presence-csv")
        for parameter in self.parameters():
            parameter.requires_grad_(mode == "all")
        if self.view_risk_head is not None:
            self.risk_head.requires_grad_(False)
        if mode == "view-risk-only":
            for parameter in self.view_risk_head.parameters():
                parameter.requires_grad_(True)
        elif mode in {"last-stage", "last-stage-lesion-aux"}:
            if mode == "last-stage-lesion-aux" and self.lesion_head is None:
                raise ValueError("last-stage-lesion-aux requires --lesion-presence-csv")
            stages = getattr(self.encoder, "stages", None)
            if stages is None or len(stages) == 0:
                raise ValueError("last-stage fine-tuning currently requires an encoder with staged blocks")
            for parameter in stages[-1].parameters():
                parameter.requires_grad_(True)
            final_norm = getattr(self.encoder, "norm", None)
            if final_norm is not None:
                for parameter in final_norm.parameters():
                    parameter.requires_grad_(True)
            for module in (self.grade_head, self.quality_head, self.lesion_head, self.view_risk_head):
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
            if self.view_risk_head is None:
                self.risk_head.requires_grad_(True)
        self.encoder_frozen = mode in {"view-risk-only", "last-stage", "last-stage-lesion-aux"}
        return {
            "mode": mode,
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "frozen_parameters": sum(p.numel() for p in self.parameters() if not p.requires_grad),
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor, view_mask: torch.Tensor, return_auxiliary: bool = False):
        batch, views, channels, height, width = images.shape
        flat_images = images.reshape(batch * views, channels, height, width)
        if self.preserve_pretrained_risk:
            encoded = self.encoder.forward_features(flat_images)
            flat_features = self.encoder.forward_head(encoded, pre_logits=True)
            pretrained_logits = self.encoder.forward_head(encoded).reshape(batch, views, -1).squeeze(-1)
        else:
            flat_features = self.encoder(flat_images)
            pretrained_logits = None
        features = flat_features.reshape(batch, views, -1)
        mask = view_mask.unsqueeze(-1).to(features.dtype)
        mean = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        maximum = features.masked_fill(~view_mask.unsqueeze(-1), torch.finfo(features.dtype).min).max(dim=1).values
        difference = torch.abs(features[:, 0] - features[:, 1]) * (view_mask[:, 1] & view_mask[:, 0]).unsqueeze(-1)
        fused = torch.cat((mean, maximum, difference), dim=1)
        if self.view_risk_head is None:
            risk = self.risk_head(fused).squeeze(1)
        else:
            per_view_risk = self.view_risk_head(features).squeeze(-1)
            risk = pool_view_risk(per_view_risk, view_mask, self.risk_pooling, self.risk_pool_temperature)
        if pretrained_logits is not None:
            prior_risk = aggregate_pretrained_view_risk(pretrained_logits, view_mask)
            risk = prior_risk + risk
        result = (self.grade_head(fused), risk, self.quality_head(fused).squeeze(1))
        if return_auxiliary:
            if self.lesion_head is None:
                raise ValueError("Auxiliary outputs requested without lesion-presence supervision")
            return (*result, self.lesion_head(fused))
        return result


def pool_view_risk(
    logits: torch.Tensor, view_mask: torch.Tensor, mode: str, temperature: float = 0.25
) -> torch.Tensor:
    """Pool view logits to an eye score without assigning labels to individual views."""
    if logits.ndim != 2 or view_mask.shape != logits.shape:
        raise ValueError("logits and view_mask must have matching [batch, views] shapes")
    if not bool(view_mask.any(dim=1).all()):
        raise ValueError("Every eye must have at least one valid view")
    masked = logits.float().masked_fill(~view_mask, torch.finfo(torch.float32).min)
    if mode == "max":
        return masked.max(dim=1).values
    if mode == "mean":
        return (masked.masked_fill(~view_mask, 0.0).sum(dim=1) / view_mask.sum(dim=1).clamp_min(1))
    if mode == "logmeanexp":
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        pooled = torch.logsumexp(masked / temperature, dim=1) * temperature
        return pooled - temperature * view_mask.sum(dim=1).float().log()
    raise ValueError(f"Unsupported view pooling mode: {mode}")


class BoundaryPairBatchSampler(Sampler[list[int]]):
    """Oversample within-source grade-1/grade-2 pairs without touching held-out data."""

    def __init__(self, items: list[dict[str, Any]], batches_per_epoch: int, seed: int) -> None:
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0
        groups: dict[tuple[str, int], list[int]] = {}
        for index, item in enumerate(items):
            groups.setdefault((item["source_dataset"], int(item["grade"])), []).append(index)
        self.boundary_sources = [
            source for source in sorted({key[0] for key in groups})
            if groups.get((source, 1)) and groups.get((source, 2))
        ]
        self.grade_zero = [idx for (source, grade), values in groups.items() if grade == 0 for idx in values]
        self.grade_severe = [idx for (source, grade), values in groups.items() if grade in (3, 4) for idx in values]
        self.ungradable = [idx for (source, grade), values in groups.items() if grade == 5 for idx in values]
        if not self.boundary_sources:
            raise ValueError("Boundary-pair sampling requires grade-1 and grade-2 eyes in at least one source")
        if not self.grade_zero or not self.grade_severe:
            raise ValueError("Sampler requires grade-0 and grade-3/4 training eyes")
        if not self.ungradable:
            raise ValueError("Sampler requires known ungradable training eyes for quality supervision")
        self.groups = groups

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for step in range(self.batches_per_epoch):
            slot = step % 10
            if slot == 8:
                # Retain a direct quality signal while the encoder is fine-tuned.
                yield [rng.choice(self.ungradable), rng.choice(self.grade_zero)]
            elif slot == 9:
                # Keep the ordinal head exposed to non-boundary grades.
                yield [rng.choice(self.grade_zero), rng.choice(self.grade_severe)]
            else:
                source = self.boundary_sources[(step // 2) % len(self.boundary_sources)]
                yield [rng.choice(self.groups[(source, 1)]), rng.choice(self.groups[(source, 2)])]

    def __len__(self) -> int:
        return self.batches_per_epoch


class HardBoundaryPairBatchSampler(Sampler[list[int]]):
    """Pair low-scored training grade-2 eyes with high-scored grade-1 eyes."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        ordinal_scores: np.ndarray,
        batches_per_epoch: int,
        seed: int,
        hard_fraction: float,
    ) -> None:
        if len(items) != len(ordinal_scores):
            raise ValueError("Training scores must align one-to-one with training eyes")
        if not 0 < hard_fraction <= 1:
            raise ValueError("hard_fraction must be in (0, 1]")
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0
        self.groups: dict[tuple[str, int], list[int]] = {}
        for index, item in enumerate(items):
            self.groups.setdefault((str(item["source_dataset"]), int(item["grade"])), []).append(index)
        self.boundary_sources = [
            source for source in sorted({key[0] for key in self.groups})
            if self.groups.get((source, 1)) and self.groups.get((source, 2))
        ]
        if not self.boundary_sources:
            raise ValueError("Hard-boundary mining requires same-source grade-1/grade-2 training pairs")

        self.hard_grade1: dict[str, list[int]] = {}
        self.hard_grade2: dict[str, list[int]] = {}
        for source in self.boundary_sources:
            grade1 = self.groups[(source, 1)]
            grade2 = self.groups[(source, 2)]
            n1 = max(1, int(np.ceil(len(grade1) * hard_fraction)))
            n2 = max(1, int(np.ceil(len(grade2) * hard_fraction)))
            self.hard_grade1[source] = sorted(grade1, key=lambda i: ordinal_scores[i], reverse=True)[:n1]
            self.hard_grade2[source] = sorted(grade2, key=lambda i: ordinal_scores[i])[:n2]
        self.grade_zero = [i for (source, grade), values in self.groups.items() if grade == 0 for i in values]
        self.grade_severe = [i for (source, grade), values in self.groups.items() if grade in (3, 4) for i in values]
        self.ungradable = [i for (source, grade), values in self.groups.items() if grade == 5 for i in values]
        if not self.grade_zero or not self.grade_severe or not self.ungradable:
            raise ValueError("Hard-boundary sampler requires grade-0, grade-3/4, and grade-5 controls")

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for step in range(self.batches_per_epoch):
            if step % 10 == 8:
                yield [rng.choice(self.ungradable), rng.choice(self.grade_zero)]
            elif step % 10 == 9:
                yield [rng.choice(self.grade_zero), rng.choice(self.grade_severe)]
            else:
                source = self.boundary_sources[(step // 2) % len(self.boundary_sources)]
                yield [rng.choice(self.hard_grade1[source]), rng.choice(self.hard_grade2[source])]

    def __len__(self) -> int:
        return self.batches_per_epoch


class BoundaryFocusedEyeBatchSampler(Sampler[list[int]]):
    """Stratify single-eye updates around grades 1/2 while retaining controls."""

    def __init__(self, items: list[dict[str, Any]], samples_per_epoch: int, seed: int) -> None:
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.epoch = 0
        groups: dict[tuple[str, int], list[int]] = {}
        for index, item in enumerate(items):
            groups.setdefault((item["source_dataset"], int(item["grade"])), []).append(index)
        self.boundary_sources = [
            source for source in sorted({key[0] for key in groups})
            if groups.get((source, 1)) and groups.get((source, 2))
        ]
        self.ungradable = [idx for (source, grade), values in groups.items() if grade == 5 for idx in values]
        self.controls = [idx for (source, grade), values in groups.items() if grade in (0, 3, 4) for idx in values]
        if not self.boundary_sources or not self.ungradable or not self.controls:
            raise ValueError("Focused sampling requires grade-1/2, ungradable, and grade-0/3/4 training eyes")
        self.groups = groups

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for step in range(self.samples_per_epoch):
            slot = step % 10
            if slot == 8:
                yield [rng.choice(self.ungradable)]
            elif slot == 9:
                yield [rng.choice(self.controls)]
            else:
                source = self.boundary_sources[(step // 4) % len(self.boundary_sources)]
                grade = 1 + (step % 2)
                yield [rng.choice(self.groups[(source, grade)])]

    def __len__(self) -> int:
        return self.samples_per_epoch


class SourceBalancedEyeBatchSampler(Sampler[list[int]]):
    """Equalize dataset exposure while preserving each source's observed grade mix."""

    def __init__(self, items: list[dict[str, Any]], batches_per_epoch: int, seed: int) -> None:
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0
        self.by_source: dict[str, list[int]] = {}
        self.by_source_grade: dict[tuple[str, int], list[int]] = {}
        for index, item in enumerate(items):
            source = str(item["source_dataset"])
            grade = int(item["grade"])
            self.by_source.setdefault(source, []).append(index)
            self.by_source_grade.setdefault((source, grade), []).append(index)
        self.sources = sorted(self.by_source)
        quality_negatives = [
            idx for (source, grade), group in self.by_source_grade.items()
            if source == "ddr_hf_specificity300" and grade < 5 for idx in group
        ]
        self.quality_pair = (self.by_source_grade.get(("ddr_hf_specificity300", 5), []), quality_negatives)
        if len(self.sources) < 2 or not self.quality_pair[0] or not self.quality_pair[1]:
            raise ValueError("Source balancing requires multiple datasets and labeled DDR quality examples")

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        bad_quality, good_quality = self.quality_pair
        for step in range(self.batches_per_epoch):
            if step % 10 == 9:
                yield [rng.choice(bad_quality), rng.choice(good_quality)]
            else:
                source = self.sources[(step // 3) % len(self.sources)]
                group = self.by_source[source]
                yield [rng.choice(group), rng.choice(group)]

    def __len__(self) -> int:
        return self.batches_per_epoch


class LesionBalancedBatchSampler(Sampler[list[int]]):
    """Guarantee one mask-supervised DDR eye in each auxiliary-learning batch."""

    def __init__(self, items: list[dict[str, Any]], batches_per_epoch: int, seed: int) -> None:
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0
        self.annotated = [index for index, item in enumerate(items) if item["lesion_known"]]
        self.unannotated = [index for index, item in enumerate(items) if not item["lesion_known"]]
        self.all_indices = list(range(len(items)))
        if not self.annotated:
            raise ValueError("Lesion-balanced sampling requires mask-supervised training eyes")
        if len(items) < 2:
            raise ValueError("Lesion-balanced sampling requires at least two training eyes")

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        labeled_cycle: list[int] = []
        for _ in range(self.batches_per_epoch):
            if not labeled_cycle:
                labeled_cycle = self.annotated.copy()
                rng.shuffle(labeled_cycle)
            annotated = labeled_cycle.pop()
            companions = self.unannotated or [index for index in self.all_indices if index != annotated]
            companion = rng.choice(companions)
            yield [annotated, companion]

    def __len__(self) -> int:
        return self.batches_per_epoch


def predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    columns: dict[str, list[Any]] = {
        key: []
        for key in ("grade", "referable", "quality_known", "ungradable", "risk", "ordinal_risk", "max_grade_margin", "grade_boundary_margin", "quality", "eye_id", "patient_id", "source_dataset", "image_ids")
    }
    with torch.inference_mode():
        for images, mask, grade, referable, quality_known, ungradable, eye_id, patient_id, source, image_ids, _, _ in loader:
            grade_logits, risk, quality = model(images.to(device), mask.to(device))
            values = {
                "grade": grade.numpy(),
                "referable": referable.numpy(),
                "quality_known": quality_known.numpy(),
                "ungradable": ungradable.numpy(),
                "risk": torch.sigmoid(risk.float()).cpu().numpy(),
                "ordinal_risk": ordinal_referable_score(grade_logits).cpu().numpy(),
                "max_grade_margin": max_grade_margin(grade_logits).cpu().numpy(),
                "grade_boundary_margin": grade_boundary_margin(grade_logits).cpu().numpy(),
                "quality": torch.sigmoid(quality.float()).cpu().numpy(),
                "eye_id": eye_id,
                "patient_id": patient_id,
                "source_dataset": source,
                "image_ids": image_ids,
            }
            for key, value in values.items():
                columns[key].extend(value.tolist() if isinstance(value, np.ndarray) else list(value))
    return {key: np.asarray(value) for key, value in columns.items()}


def filter_prediction_sources(
    scores: dict[str, np.ndarray], sources: list[str] | None
) -> dict[str, np.ndarray]:
    if not sources:
        return scores
    mask = np.isin(scores["source_dataset"].astype(str), sources)
    if not mask.any():
        raise ValueError(f"No prediction rows match evaluation sources: {sources}")
    return {key: value[mask] for key, value in scores.items()}


def select_filtered_policy(
    scores: dict[str, np.ndarray], sources: list[str] | None, score_key: str = "risk"
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    filtered = filter_prediction_sources(scores, sources)
    policy = select_policy(
        {
            "grade": filtered["grade"],
            "risk": filtered[score_key],
            "quality": filtered["quality"],
            "quality_known": filtered["quality_known"],
            "ungradable": filtered["ungradable"],
        }
    )
    return filtered, policy


def ordinal_referable_score(grade_logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(grade_logits.float(), dim=1)[:, 2:5].sum(dim=1)


def max_grade_margin(grade_logits: torch.Tensor) -> torch.Tensor:
    """Max referable-grade logit minus max nonreferable-grade logit."""
    if grade_logits.ndim != 2 or grade_logits.shape[1] != 5:
        raise ValueError("Expected five grade logits for grades 0-4")
    logits = grade_logits.float()
    return logits[:, 2:5].max(dim=1).values - logits[:, :2].max(dim=1).values


def grade_boundary_margin(grade_logits: torch.Tensor) -> torch.Tensor:
    """Grade-2 logit minus grade-1 logit, targeting the referable boundary."""
    if grade_logits.ndim != 2 or grade_logits.shape[1] != 5:
        raise ValueError("Expected five grade logits for grades 0-4")
    return grade_logits.float()[:, 2] - grade_logits.float()[:, 1]


def ordinal_referable_logit(grade_logits: torch.Tensor) -> torch.Tensor:
    logits = grade_logits.float()
    positive = torch.logsumexp(logits[:, 2:5], dim=1)
    nonreferable_or_ungradable = torch.logsumexp(
        torch.cat((logits[:, :2], logits[:, 5:]), dim=1), dim=1
    )
    return positive - nonreferable_or_ungradable


def cumulative_ordinal_loss(
    grade_logits: torch.Tensor, grade: torch.Tensor, gradable: torch.Tensor
) -> torch.Tensor:
    logits = grade_logits.float()[gradable]
    targets = grade[gradable]
    if not len(targets):
        return grade_logits.sum() * 0.0
    boundaries = torch.arange(1, grade_logits.shape[1], device=targets.device)
    cumulative_logits = torch.stack([
        torch.logsumexp(logits[:, boundary:], dim=1)
        - torch.logsumexp(logits[:, :boundary], dim=1)
        for boundary in range(1, grade_logits.shape[1])
    ], dim=1)
    cumulative_targets = (targets[:, None] >= boundaries[None, :]).float()
    return nn.functional.binary_cross_entropy_with_logits(cumulative_logits, cumulative_targets)


def aggregate_pretrained_view_risk(logits: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~view_mask, torch.finfo(logits.dtype).min).max(dim=1).values


def select_manifest_sources(
    rows: list[dict[str, Any]], sources: list[str] | None, split: str
) -> list[dict[str, Any]]:
    if not sources:
        return rows
    selected = [row for row in rows if str(row["source_dataset"]) in sources]
    if not selected:
        raise ValueError(f"No {split} manifest rows match sources: {sources}")
    return selected


def manifest_view_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    per_eye: dict[str, int] = {}
    for row in rows:
        eye_id = str(row["eye_id"])
        per_eye[eye_id] = per_eye.get(eye_id, 0) + 1
    return {
        "eyes": len(per_eye),
        "one_view": sum(count == 1 for count in per_eye.values()),
        "two_view": sum(count == 2 for count in per_eye.values()),
    }


def stratified_policy_metrics(
    scores: dict[str, np.ndarray], policy: dict[str, Any], score_key: str = "risk"
) -> dict[str, dict[str, Any]]:
    """Report the same validation-selected policy by source and reference grade."""
    output: dict[str, dict[str, Any]] = {}
    base = {key: scores[key] for key in ("grade", "risk", "quality", "quality_known", "ungradable")}
    base["risk"] = scores[score_key]
    sources = scores.get("source_dataset")
    if sources is not None:
        for source in sorted(set(sources.tolist())):
            mask = sources == source
            output[f"source:{source}"] = evaluate_policy(
                {key: value[mask] for key, value in base.items()}, policy
            )
    grades = scores["grade"].astype(int)
    for grade in sorted(set(grades.tolist())):
        mask = grades == grade
        output[f"grade:{grade}"] = evaluate_policy(
            {key: value[mask] for key, value in base.items()}, policy
        )
    return output


def write_predictions(
    path: Path, split: str, scores: dict[str, np.ndarray], policy: dict[str, Any], score_key: str = "risk"
) -> None:
    referred = (scores[score_key].astype(float) >= float(policy["risk_threshold"])) | (
        scores["quality"].astype(float) >= float(policy["quality_threshold"])
    )
    keys = ["eye_id", "patient_id", "source_dataset", "grade", "referable", "quality_known", "ungradable", "risk", "ordinal_risk", "max_grade_margin", "grade_boundary_margin", "quality"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", *keys, "image_ids", "refer_action"])
        writer.writeheader()
        for i in range(len(scores["grade"])):
            row = {key: scores[key][i].item() if isinstance(scores[key][i], np.generic) else scores[key][i] for key in keys}
            row.update(split=split, image_ids=scores["image_ids"][i], refer_action=int(referred[i]))
            writer.writerow(row)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--model-name", default="convnext_tiny.fb_in22k_ft_in1k")
    parser.add_argument("--preserve-pretrained-risk", action="store_true",
                        help="Use the pretrained scalar disease head as a risk prior and learn a zero-initialized residual.")
    parser.add_argument("--risk-pooling", choices=("fused", "mean", "max", "logmeanexp"), default="fused",
                        help="Use an eye-level multiple-instance disease head over view embeddings; view labels are never fabricated.")
    parser.add_argument("--risk-pool-temperature", type=float, default=0.25)
    parser.add_argument("--selection-score", choices=("risk", "ordinal_risk", "max_grade_margin", "grade_boundary_margin"), default="risk",
                        help="Validation score used to select the checkpoint and zero-miss policy.")
    parser.add_argument("--lesion-presence-csv",
                        help="Optional DDR train-only mask-derived EX/HE/MA/SE labels keyed by image_basename.")
    parser.add_argument("--lesion-loss-weight", type=float, default=0.5)
    parser.add_argument("--debug-finite-steps", action="store_true",
                        help="Check model outputs, gradients, and parameters after each step for numerical failures.")
    parser.add_argument("--max-train-batches", type=int, default=0,
                        help="Diagnostic-only batch cap; any resulting checkpoint is not a trained candidate.")
    parser.add_argument("--amp-init-scale", type=float, default=65536.0,
                        help="Initial CUDA AMP loss scale; reduce for diagnosing repeated overflow skips.")
    parser.add_argument("--lesion-aux-batches", type=int, default=0,
                        help="Use this many two-eye batches per epoch, each containing one mask-labeled DDR eye.")
    parser.add_argument("--trainable-components", choices=("all", "view-risk-only", "last-stage", "last-stage-lesion-aux"), default="all",
                        help="Choose full, risk-head-only, last-stage, or last-stage plus lesion-auxiliary fine-tuning.")
    parser.add_argument("--rank-score", choices=("binary", "ordinal"), default="binary",
                        help="Train the pairwise boundary objective on the binary-risk head or the deployed ordinal risk score.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--boundary-pair-batches", type=int, default=0,
                        help="Use source-matched grade-1/grade-2 batches; 0 keeps ordinary shuffled sampling.")
    parser.add_argument("--hard-boundary-batches", type=int, default=0,
                        help="Mine hard grade-1/2 pairs using scores computed only on the training split.")
    parser.add_argument("--hard-boundary-fraction", type=float, default=0.25)
    parser.add_argument("--boundary-focused-samples", type=int, default=0,
                        help="Use stratified one-eye updates around grades 1/2; requires batch size 1.")
    parser.add_argument("--source-balanced-batches", type=int, default=0,
                        help="Equalize source exposure with natural within-source grades; requires batch size 2.")
    parser.add_argument("--evaluation-sources", nargs="+", default=None,
                        help="Restrict policy selection and aggregate metrics to these source_dataset values.")
    parser.add_argument("--training-sources", nargs="+", default=None,
                        help="Restrict training to these source_dataset values; held-out splits remain patient-disjoint.")
    parser.add_argument("--skip-test-evaluation", action="store_true",
                        help="Do not load or score development-test images; select this candidate using validation only.")
    parser.add_argument("--rank-weight", type=float, default=0.25)
    parser.add_argument("--cumulative-ordinal-weight", type=float, default=0.0,
                        help="Add a cumulative ordinal loss over every grade boundary, including grade 1 vs 2.")
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument("--rank-top-fraction", type=float, default=0.25)
    parser.add_argument("--grade2-rank-weight", type=float, default=1.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()

    if args.lesion_loss_weight < 0:
        raise ValueError("--lesion-loss-weight must be nonnegative")
    if args.cumulative_ordinal_weight < 0:
        raise ValueError("--cumulative-ordinal-weight must be nonnegative")
    if args.max_train_batches < 0:
        raise ValueError("--max-train-batches must be nonnegative")
    if args.amp_init_scale <= 0:
        raise ValueError("--amp-init-scale must be positive")
    if args.lesion_aux_batches < 0:
        raise ValueError("--lesion-aux-batches must be nonnegative")
    if args.hard_boundary_batches < 0:
        raise ValueError("--hard-boundary-batches must be nonnegative")
    if not 0 < args.hard_boundary_fraction <= 1:
        raise ValueError("--hard-boundary-fraction must be in (0, 1]")

    seed_all(args.seed)
    root = Path.cwd()
    splits = read_rows(Path(args.manifest))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # A warm start already supplies encoder weights; avoid a needless hub lookup.
    lesion_presence: dict[str, tuple[int, int, int, int]] = {}
    if args.lesion_presence_csv:
        with Path(args.lesion_presence_csv).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                filename = str(row["image_basename"])
                if filename in lesion_presence:
                    raise ValueError(f"Duplicate lesion-presence label for {filename}")
                target = tuple(int(row[name]) for name in ("ex", "he", "ma", "se"))
                if any(value not in (0, 1) for value in target):
                    raise ValueError(f"Lesion-presence targets must be binary: {filename}")
                lesion_presence[filename] = target
        training_ddr_names = {
            Path(row["image_path"]).name for row in splits["train"]
            if row.get("source_dataset") == "ddr_hf_specificity300"
        }
        heldout_names = {
            Path(row["image_path"]).name for split in ("valid", "test") for row in splits[split]
            if row.get("source_dataset") == "ddr_hf_specificity300"
        }
        if not set(lesion_presence).issubset(training_ddr_names):
            raise ValueError("Lesion labels include images outside the DDR training partition")
        if set(lesion_presence) & heldout_names:
            raise ValueError("Lesion labels overlap a held-out DDR partition")
    model = MixedViewModel(
        args.model_name, pretrained=not bool(args.init_checkpoint), image_size=args.image_size,
        preserve_pretrained_risk=args.preserve_pretrained_risk,
        risk_pooling=args.risk_pooling, risk_pool_temperature=args.risk_pool_temperature,
        lesion_presence_classes=4 if args.lesion_presence_csv else 0,
    ).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("model_name") != args.model_name:
            raise ValueError("Warm-start checkpoint backbone does not match --model-name")
        incompatible = model.load_state_dict(
            initial["state_dict"], strict=args.risk_pooling == "fused" and not args.lesion_presence_csv
        )
        if args.risk_pooling != "fused":
            allowed_missing = ("view_risk_head.", "lesion_head.") if args.lesion_presence_csv else ("view_risk_head.",)
            if incompatible.unexpected_keys or any(not key.startswith(allowed_missing) for key in incompatible.missing_keys):
                raise ValueError(
                    "Warm-start checkpoint is structurally incompatible; only declared new heads may be missing. "
                    f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
                )
        elif args.lesion_presence_csv and (incompatible.unexpected_keys or any(
            not key.startswith("lesion_head.") for key in incompatible.missing_keys
        )):
            raise ValueError(
                "Warm-start checkpoint is structurally incompatible; only the auxiliary lesion head may be missing. "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
    trainable_info = model.configure_trainable_components(args.trainable_components)
    if args.trainable_components == "last-stage-lesion-aux" and not args.lesion_presence_csv:
        raise ValueError("last-stage-lesion-aux requires --lesion-presence-csv")
    train_rows = select_manifest_sources(splits["train"], args.training_sources, "train")
    valid_rows = select_manifest_sources(splits["valid"], args.evaluation_sources, "validation")
    test_rows = select_manifest_sources(splits["test"], args.evaluation_sources, "test")
    train_data = EyeBags(train_rows, model.train_transform, root, lesion_presence)
    valid_data = EyeBags(valid_rows, model.eval_transform, root)
    test_data = None if args.skip_test_evaluation else EyeBags(test_rows, model.eval_transform, root)
    pin = device.type == "cuda"
    sampler_modes = sum(bool(value) for value in (
        args.boundary_pair_batches, args.hard_boundary_batches,
        args.boundary_focused_samples, args.source_balanced_batches, args.lesion_aux_batches
    ))
    if sampler_modes > 1:
        raise ValueError("Choose only one explicit sampling mode")
    hard_sampler: HardBoundaryPairBatchSampler | None = None
    if args.lesion_aux_batches:
        if not args.lesion_presence_csv:
            raise ValueError("--lesion-aux-batches requires --lesion-presence-csv")
        if args.trainable_components != "last-stage-lesion-aux":
            raise ValueError("--lesion-aux-batches requires --trainable-components last-stage-lesion-aux")
        if args.batch_size != 2:
            raise ValueError("--lesion-aux-batches requires --batch-size 2")
        sampler = LesionBalancedBatchSampler(train_data.items, args.lesion_aux_batches, args.seed)
        train_loader = DataLoader(train_data, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=pin)
    elif args.hard_boundary_batches:
        if args.batch_size != 2:
            raise ValueError("--hard-boundary-batches requires --batch-size 2")
        train_scoring_rows = [row for row in train_rows if int(row["grade"]) in (1, 2)]
        train_scoring_data = EyeBags(train_scoring_rows, model.eval_transform, root)
        train_scoring_loader = DataLoader(
            train_scoring_data, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=pin,
        )
        scored = predictions(model, train_scoring_loader, device)
        train_item_indices = {item["eye_id"]: index for index, item in enumerate(train_data.items)}
        train_scores = np.zeros(len(train_data.items), dtype=np.float32)
        for eye_id, score in zip(scored["eye_id"], scored["ordinal_risk"]):
            if str(eye_id) not in train_item_indices:
                raise ValueError(f"Hard-mining score is not from the training partition: {eye_id}")
            train_scores[train_item_indices[str(eye_id)]] = float(score)
        expected_boundary_ids = {
            item["eye_id"] for item in train_data.items if int(item["grade"]) in (1, 2)
        }
        if set(scored["eye_id"].tolist()) != expected_boundary_ids:
            raise ValueError("Hard-mining scores do not cover exactly the training grade-1/grade-2 eyes")
        hard_sampler = HardBoundaryPairBatchSampler(
            train_data.items, train_scores, args.hard_boundary_batches, args.seed, args.hard_boundary_fraction
        )
        train_loader = DataLoader(train_data, batch_sampler=hard_sampler, num_workers=args.num_workers, pin_memory=pin)
    elif args.source_balanced_batches:
        if args.batch_size != 2:
            raise ValueError("--source-balanced-batches requires --batch-size 2")
        sampler = SourceBalancedEyeBatchSampler(train_data.items, args.source_balanced_batches, args.seed)
        train_loader = DataLoader(train_data, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=pin)
    elif args.boundary_focused_samples:
        if args.batch_size != 1:
            raise ValueError("--boundary-focused-samples requires --batch-size 1")
        sampler = BoundaryFocusedEyeBatchSampler(train_data.items, args.boundary_focused_samples, args.seed)
        train_loader = DataLoader(train_data, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=pin)
    elif args.boundary_pair_batches:
        if args.batch_size != 2:
            raise ValueError("--boundary-pair-batches requires --batch-size 2")
        batch_sampler = BoundaryPairBatchSampler(train_data.items, args.boundary_pair_batches, args.seed)
        train_loader = DataLoader(train_data, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=pin)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    test_loader = None if test_data is None else DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin
    )
    optimizer_groups = []
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder_params:
        optimizer_groups.append({"params": encoder_params, "lr": args.backbone_lr})
    head_params = [p for module in (model.grade_head, model.risk_head, model.view_risk_head, model.quality_head, model.lesion_head)
                   if module is not None for p in module.parameters() if p.requires_grad]
    if head_params:
        optimizer_groups.append({"params": head_params, "lr": args.head_lr})
    if not optimizer_groups:
        raise ValueError("The selected trainable component mode produced no optimizer parameters")
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=args.amp_init_scale)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    # Treat the supplied warm start (or pretrained initialization) as a real
    # candidate. Fine-tuning must beat it on the same validation policy before
    # it can replace the checkpoint.
    initial_pred = predictions(model, valid_loader, device)
    initial_policy_pred, initial_policy = select_filtered_policy(
        initial_pred, args.evaluation_sources, args.selection_score
    )
    _, initial_ordinal_policy = select_filtered_policy(initial_pred, args.evaluation_sources, "ordinal_risk")
    initial_record = {
        "epoch": 0,
        "selection_score": args.selection_score,
        "risk_pooling": args.risk_pooling,
        "trainable_components": args.trainable_components,
        "train_loss": None,
        "validation_policy": initial_policy,
        "validation_metrics": evaluate_policy(
            {"grade": initial_policy_pred["grade"], "risk": initial_policy_pred[args.selection_score],
             "quality": initial_policy_pred["quality"], "quality_known": initial_policy_pred["quality_known"],
             "ungradable": initial_policy_pred["ungradable"]}, initial_policy
        ),
        "validation_metrics_by_source_and_grade": stratified_policy_metrics(
            initial_policy_pred, initial_policy, args.selection_score
        ),
        "ordinal_validation_policy": initial_ordinal_policy,
        "ordinal_validation_metrics": evaluate_policy(
            {"grade": initial_policy_pred["grade"], "risk": initial_policy_pred["ordinal_risk"],
             "quality": initial_policy_pred["quality"], "quality_known": initial_policy_pred["quality_known"],
             "ungradable": initial_policy_pred["ungradable"]}, initial_ordinal_policy
        ),
    }
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_record = copy.deepcopy(initial_record)
    best_key = (
        float(initial_policy["specificity_gradable_nonreferable"]),
        -float(initial_policy["referral_rate"]),
    )
    history = [initial_record]
    Path(args.artifact).parent.mkdir(parents=True, exist_ok=True)
    inference_state = {key: value for key, value in best_state.items() if not key.startswith("lesion_head.")}
    torch.save({"model_name": args.model_name, "state_dict": inference_state, "best_epoch": best_record,
                "args": vars(args), "claim_boundary": "DEVELOPMENT_CANDIDATE_NOT_CERTIFIED"}, args.artifact)
    (out / "progress.json").write_text(
        json.dumps({"status": "TRAINING_IN_PROGRESS", "history": history}, indent=2) + "\n", encoding="utf-8"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_examples = 0.0, 0
        skipped_nonfinite_gradient_batches = 0
        gradient_skip_details: list[dict[str, Any]] = []
        for step, (images, mask, grade, referable, quality_known, ungradable, *batch_tail) in enumerate(train_loader, 1):
            lesion_targets, lesion_known = batch_tail[-2:]
            images = images.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            grade = grade.to(device)
            referable = referable.float().to(device)
            quality_known = quality_known.bool().to(device)
            ungradable = ungradable.float().to(device)
            lesion_targets = lesion_targets.to(device)
            lesion_known = lesion_known.bool().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                outputs = model(images, mask, return_auxiliary=bool(args.lesion_presence_csv))
                grade_logits, risk_logits, quality_logits = outputs[:3]
                if args.debug_finite_steps and not all(bool(torch.isfinite(value).all()) for value in outputs):
                    bad_parameters = [
                        name for name, parameter in model.named_parameters()
                        if not bool(torch.isfinite(parameter).all())
                    ]
                    raise FloatingPointError(
                        f"Non-finite model output before loss at epoch {epoch}, step {step}; "
                        f"inputs_finite={bool(torch.isfinite(images).all())}, "
                        f"bad_parameters={bad_parameters[:12]}, image_ids={list(batch_tail[3])}"
                    )
                gradable = grade < 5
                grade_loss = masked_cross_entropy(grade_logits, grade, gradable)
                cumulative_loss = cumulative_ordinal_loss(grade_logits, grade, gradable)
                risk_loss = masked_bce_with_logits(risk_logits, referable, gradable)
                quality_loss = masked_bce_with_logits(quality_logits, ungradable, quality_known)
                rank_scores = ordinal_referable_logit(grade_logits) if args.rank_score == "ordinal" else risk_logits
                rank_loss = hard_boundary_rank_loss(
                    rank_scores, grade, referable, gradable,
                    args.rank_margin, args.rank_top_fraction, args.grade2_rank_weight,
                )
                lesion_loss = (
                    masked_bce_with_logits(outputs[3], lesion_targets, lesion_known)
                    if args.lesion_presence_csv else risk_logits.sum() * 0.0
                )
                loss = (grade_loss + risk_loss + quality_loss
                        + args.cumulative_ordinal_weight * cumulative_loss
                        + args.rank_weight * rank_loss)
                loss = loss + args.lesion_loss_weight * lesion_loss
            if not torch.isfinite(loss):
                components = {
                    "grade": float(grade_loss.detach().float().cpu()),
                    "cumulative_ordinal": float(cumulative_loss.detach().float().cpu()),
                    "risk": float(risk_loss.detach().float().cpu()),
                    "quality": float(quality_loss.detach().float().cpu()),
                    "rank": float(rank_loss.detach().float().cpu()),
                    "lesion": float(lesion_loss.detach().float().cpu()),
                    "finite_outputs": [bool(torch.isfinite(value).all()) for value in outputs],
                    "lesion_known": lesion_known.detach().cpu().tolist(),
                    "lesion_targets": lesion_targets.detach().cpu().tolist(),
                    "image_ids": list(batch_tail[3]),
                }
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {json.dumps(components, sort_keys=True)}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            bad_gradients = [
                name for name, parameter in model.named_parameters()
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            ] if args.debug_finite_steps else []
            if bad_gradients:
                if args.debug_finite_steps and len(gradient_skip_details) < 32:
                    gradient_skip_details.append({
                        "step": step,
                        "image_ids": list(batch_tail[3]),
                        "source_dataset": list(batch_tail[2]),
                        "grade": grade.detach().cpu().tolist(),
                        "loss": float(loss.detach().float().cpu()),
                        "components": {
                            "grade": float(grade_loss.detach().float().cpu()),
                            "cumulative_ordinal": float(cumulative_loss.detach().float().cpu()),
                            "risk": float(risk_loss.detach().float().cpu()),
                            "quality": float(quality_loss.detach().float().cpu()),
                            "rank": float(rank_loss.detach().float().cpu()),
                            "lesion": float(lesion_loss.detach().float().cpu()),
                        },
                        "scale_before_update": scaler.get_scale(),
                        "bad_gradient_parameters": bad_gradients[:12],
                    })
                scaler.step(optimizer)
                scaler.update()
                skipped_nonfinite_gradient_batches += 1
                continue
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            bad_parameters = [
                name for name, parameter in model.named_parameters()
                if not bool(torch.isfinite(parameter).all())
            ] if args.debug_finite_steps else []
            if bad_parameters:
                raise FloatingPointError(
                    f"Optimizer created non-finite parameters at epoch {epoch}, step {step}; "
                    f"loss={float(loss.detach().float().cpu())}, "
                    f"lesion_loss={float(lesion_loss.detach().float().cpu())}, "
                    f"scale={scaler.get_scale()}, bad_parameters={bad_parameters[:12]}, "
                    f"image_ids={list(batch_tail[3])}"
                )
            total_loss += float(loss.detach().cpu()) * len(grade)
            n_examples += len(grade)
            if args.max_train_batches and step >= args.max_train_batches:
                break

        valid_pred = predictions(model, valid_loader, device)
        policy_valid_pred, policy = select_filtered_policy(
            valid_pred, args.evaluation_sources, args.selection_score
        )
        _, ordinal_policy = select_filtered_policy(valid_pred, args.evaluation_sources, "ordinal_risk")
        record = {
            "epoch": epoch,
            "selection_score": args.selection_score,
            "rank_score": args.rank_score,
            "cumulative_ordinal_loss_weight": args.cumulative_ordinal_weight,
            "risk_pooling": args.risk_pooling,
            "trainable_components": args.trainable_components,
            "train_loss": total_loss / max(1, n_examples),
            "train_batch_count": step,
            "partial_epoch_diagnostic": bool(args.max_train_batches),
            "skipped_nonfinite_gradient_batches": skipped_nonfinite_gradient_batches,
            "gradient_skip_details": gradient_skip_details if args.debug_finite_steps else [],
            "validation_policy": policy,
            "validation_metrics": evaluate_policy(
                {"grade": policy_valid_pred["grade"], "risk": policy_valid_pred[args.selection_score], "quality": policy_valid_pred["quality"],
                 "quality_known": policy_valid_pred["quality_known"], "ungradable": policy_valid_pred["ungradable"]}, policy
            ),
            "validation_metrics_by_source_and_grade": stratified_policy_metrics(
                policy_valid_pred, policy, args.selection_score
            ),
            "ordinal_validation_policy": ordinal_policy,
            "ordinal_validation_metrics": evaluate_policy(
                {"grade": policy_valid_pred["grade"], "risk": policy_valid_pred["ordinal_risk"], "quality": policy_valid_pred["quality"],
                 "quality_known": policy_valid_pred["quality_known"], "ungradable": policy_valid_pred["ungradable"]}, ordinal_policy
            ),
        }
        history.append(record)
        if args.hard_boundary_batches:
            if hard_sampler is None:
                raise RuntimeError("Hard-boundary sampler was not initialized")
            record["training_sampler"] = "train_only_hard_grade1_grade2_pairs"
            record["hard_boundary_batches"] = args.hard_boundary_batches
            record["hard_boundary_fraction"] = args.hard_boundary_fraction
            record["mined_training_eyes_by_source"] = {
                source: {
                    "grade1": len(hard_sampler.hard_grade1[source]),
                    "grade2": len(hard_sampler.hard_grade2[source]),
                }
                for source in hard_sampler.boundary_sources
            }
        elif args.boundary_pair_batches:
            record["training_sampler"] = "source_matched_grade1_grade2_with_quality_and_extreme_grade_batches"
            record["boundary_pair_batches"] = args.boundary_pair_batches
        elif args.boundary_focused_samples:
            record["training_sampler"] = "stratified_grade1_grade2_single_eye_updates_with_quality_and_extreme_grade_controls"
            record["boundary_focused_samples"] = args.boundary_focused_samples
        elif args.source_balanced_batches:
            record["training_sampler"] = "equal_source_exposure_natural_within_source_grades_with_quality_controls"
            record["source_balanced_batches"] = args.source_balanced_batches
        key = (float(policy["specificity_gradable_nonreferable"]), -float(policy["referral_rate"]))
        if key > best_key:
            best_key = key
            best_record = copy.deepcopy(record)
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            Path(args.artifact).parent.mkdir(parents=True, exist_ok=True)
            inference_state = {key: value for key, value in best_state.items() if not key.startswith("lesion_head.")}
            torch.save({"model_name": args.model_name, "state_dict": inference_state, "best_epoch": best_record,
                        "args": vars(args), "claim_boundary": "DEVELOPMENT_CANDIDATE_NOT_CERTIFIED"}, args.artifact)
        (out / "progress.json").write_text(json.dumps({"status": "TRAINING_IN_PROGRESS", "history": history}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, sort_keys=True), flush=True)

    if best_state is None or best_record is None:
        raise RuntimeError("Training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    valid_pred = predictions(model, valid_loader, device)
    policy_valid_pred, policy = select_filtered_policy(
        valid_pred, args.evaluation_sources, args.selection_score
    )
    _, ordinal_policy = select_filtered_policy(valid_pred, args.evaluation_sources, "ordinal_risk")
    valid_input = {key: policy_valid_pred[key] for key in ("grade", "quality", "quality_known", "ungradable")}
    valid_input["risk"] = policy_valid_pred[args.selection_score]
    ordinal_valid_input = {key: policy_valid_pred[key] for key in ("grade", "quality", "quality_known", "ungradable")}
    ordinal_valid_input["risk"] = policy_valid_pred["ordinal_risk"]
    write_predictions(out / "validation_predictions.csv", "valid", valid_pred, policy, args.selection_score)
    if test_loader is None:
        test_metrics = None
        ordinal_test_metrics = None
    else:
        test_pred = predictions(model, test_loader, device)
        policy_test_pred = filter_prediction_sources(test_pred, args.evaluation_sources)
        test_input = {key: policy_test_pred[key] for key in ("grade", "quality", "quality_known", "ungradable")}
        test_input["risk"] = policy_test_pred[args.selection_score]
        ordinal_test_input = {key: policy_test_pred[key] for key in ("grade", "quality", "quality_known", "ungradable")}
        ordinal_test_input["risk"] = policy_test_pred["ordinal_risk"]
        test_metrics = evaluate_policy(test_input, policy)
        ordinal_test_metrics = evaluate_policy(ordinal_test_input, ordinal_policy)
        write_predictions(out / "test_predictions_development_only.csv", "test", test_pred, policy, args.selection_score)
    summary = {
        "status": "PARTIAL_TRAINING_DIAGNOSTIC_NOT_CANDIDATE" if args.max_train_batches else "DEVELOPMENT_CANDIDATE_NOT_CERTIFIED",
        "claim_boundary": "The mixed manifest has been used for prior development and is not an untouched certification cohort.",
        "unit_of_analysis": "eye-level bag; one DDR image or paired DRTiD/DeepDRiD macula+optic-disc images",
        "manifest": args.manifest,
        "model_name": args.model_name,
        "risk_pooling": args.risk_pooling,
        "risk_pool_temperature": args.risk_pool_temperature,
        "selection_score": args.selection_score,
        "rank_score": args.rank_score,
        "cumulative_ordinal_loss_weight": args.cumulative_ordinal_weight,
        "trainable_components": trainable_info,
        "training_sources": args.training_sources,
        "evaluation_sources": args.evaluation_sources,
        "best_epoch": best_record,
        "selected_validation_policy": policy,
        "validation_metrics": evaluate_policy(valid_input, policy),
        "validation_metrics_by_source_and_grade": stratified_policy_metrics(
            policy_valid_pred, policy, args.selection_score
        ),
        "test_metrics_development_only": test_metrics,
        "selected_ordinal_validation_policy": ordinal_policy,
        "ordinal_validation_metrics": evaluate_policy(ordinal_valid_input, ordinal_policy),
        "ordinal_test_metrics_development_only": ordinal_test_metrics,
        "train_eyes": len(train_data), "validation_eyes": len(valid_data), "test_eyes": len(test_rows),
        "test_evaluation_skipped": args.skip_test_evaluation,
        "partial_training_diagnostic": bool(args.max_train_batches),
        "view_counts": {
            split: {"eyes": len(dataset), "one_view": sum(len(x["views"]) == 1 for x in dataset.items),
                   "two_view": sum(len(x["views"]) == 2 for x in dataset.items)}
            for split, dataset in (("train", train_data), ("valid", valid_data))
        } | {"test": manifest_view_counts(test_rows)},
        "quality_label_boundary": "DDR grade 5 is ungradable; DRTiD quality labels are unknown and excluded from quality loss.",
        "lesion_presence_supervision": {
            "enabled": bool(args.lesion_presence_csv),
            "label_csv": args.lesion_presence_csv,
            "supervised_training_eyes": sum(int(item["lesion_known"]) for item in train_data.items),
            "lesions": ["EX", "HE", "MA", "SE"] if args.lesion_presence_csv else [],
            "loss_weight": args.lesion_loss_weight if args.lesion_presence_csv else 0.0,
            "auxiliary_head_exported": False,
        },
        "history": history,
        "artifact": args.artifact,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "progress.json").write_text(json.dumps({"status": "COMPLETE", "epochs": args.epochs}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

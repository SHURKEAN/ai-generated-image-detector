"""Manifest-backed datasets with label-independent image processing."""

from __future__ import annotations

import random
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .manifest import ManifestRecord, PROJECT_ROOT, resolve_image_path


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
LABEL_TO_INDEX = {"real": 0, "ai": 1}


class RandomJPEG:
    """Apply the same random recompression policy to both real and AI images."""

    def __init__(self, probability: float = 0.25, min_quality: int = 55) -> None:
        self.probability = probability
        self.min_quality = min_quality

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(self.min_quality, 95))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


def training_transform() -> Callable[[Image.Image], Tensor]:
    """Moderate, label-independent augmentation for real-world robustness."""

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.70, 1.0),
                ratio=(0.85, 1.18),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08)],
                p=0.25,
            ),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.15),
            RandomJPEG(probability=0.25, min_quality=55),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def evaluation_transform() -> Callable[[Image.Image], Tensor]:
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def records_for_track(
    records: Sequence[ManifestRecord],
    *,
    split: str,
    track: str,
) -> list[ManifestRecord]:
    if track == "product_safe":
        allowed_tracks = {"product_safe"}
    elif track == "academic":
        allowed_tracks = {"product_safe", "academic_only"}
    elif track == "evaluation":
        allowed_tracks = {"product_safe", "academic_only", "private_evaluation"}
    else:
        raise ValueError("track must be product_safe, academic, or evaluation")
    return [
        record
        for record in records
        if record.split == split and record.usage_track in allowed_tracks
    ]


class ManifestImageDataset(Dataset[tuple[Tensor, Tensor, str, str]]):
    def __init__(
        self,
        records: Sequence[ManifestRecord],
        *,
        project_root: Path = PROJECT_ROOT,
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        if not records:
            raise ValueError("No eligible records were selected for this dataset.")
        self.records = list(records)
        self.project_root = project_root
        self.transform = transform or evaluation_transform()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str, str]:
        record = self.records[index]
        path = resolve_image_path(record, self.project_root)
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                tensor = self.transform(image)
        except (OSError, SyntaxError, ValueError) as exc:
            raise RuntimeError(f"Could not decode manifest image {path}") from exc
        label = torch.tensor(float(LABEL_TO_INDEX[record.label]), dtype=torch.float32)
        return tensor, label, record.source_id, record.image_id


def source_balanced_sampler(records: Sequence[ManifestRecord], seed: int = 2026) -> WeightedRandomSampler:
    """Balance labels first, then distribute each label equally across sources."""

    group_counts = Counter((record.label, record.source_id) for record in records)
    sources_per_label: dict[str, int] = {
        label: len({record.source_id for record in records if record.label == label})
        for label in {record.label for record in records}
    }
    weights = [
        1.0
        / (
            sources_per_label[record.label]
            * group_counts[(record.label, record.source_id)]
        )
        for record in records
    ]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(records),
        replacement=True,
        generator=generator,
    )


__all__ = [
    "CLIP_MEAN",
    "CLIP_STD",
    "LABEL_TO_INDEX",
    "ManifestImageDataset",
    "RandomJPEG",
    "evaluation_transform",
    "records_for_track",
    "source_balanced_sampler",
    "training_transform",
]

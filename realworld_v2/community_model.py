"""Loader for the official CVPR 2025 Community Forensics checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import timm
import torch
from safetensors.torch import load_file
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "realworld_v2" / "upstream" / "commfor-model-384.safetensors"
)
CONFIG_PATH = (
    PROJECT_ROOT / "models" / "realworld_v2" / "upstream" / "commfor-model-384.config.json"
)
EXPECTED_SHA256 = "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387"
MODEL_REVISION = "6076002bf0d9dd37537f965ee2f06f826c333b61"


def community_transform() -> transforms.Compose:
    """Exact test-time geometry and normalization from the official repository."""

    return transforms.Compose(
        [
            transforms.Resize(440, interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def community_assets_available() -> bool:
    return CHECKPOINT_PATH.is_file() and CONFIG_PATH.is_file()


def load_community_model(
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    if not community_assets_available():
        raise FileNotFoundError(
            "Community Forensics assets are missing under models/realworld_v2/upstream."
        )
    if _sha256(CHECKPOINT_PATH) != EXPECTED_SHA256:
        raise RuntimeError("Community Forensics checkpoint failed SHA-256 verification.")

    # The complete fine-tuned state is in safetensors, so pretrained=False avoids
    # a hidden network download and is exactly replaced by the official weights.
    model = timm.create_model(
        "vit_small_patch16_384.augreg_in21k_ft_in1k",
        pretrained=False,
        num_classes=1,
    )
    wrapped_state = load_file(str(CHECKPOINT_PATH), device="cpu")
    state = {
        key.removeprefix("vit."): value
        for key, value in wrapped_state.items()
    }
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    metadata = {
        "model_id": "community_forensics_cvpr2025",
        "version": MODEL_REVISION,
        "architecture": "ViT-S/16 384",
        "training_scope": "4,803 generator models",
        "usage_track": "academic_reference",
        "threshold": 0.5,
    }
    return model, metadata


__all__ = [
    "CHECKPOINT_PATH",
    "EXPECTED_SHA256",
    "community_assets_available",
    "community_transform",
    "load_community_model",
]


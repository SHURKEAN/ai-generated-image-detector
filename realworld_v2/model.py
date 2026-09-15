"""Exact CLIP ViT-L/14 backbone and binary head used by real-world v2."""

from __future__ import annotations

import importlib.util
import hashlib
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import Tensor, nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_MODEL_PATH = (
    PROJECT_ROOT / "third_party" / "UniversalFakeDetect" / "models" / "clip" / "model.py"
)
DEFAULT_BACKBONE_PATH = PROJECT_ROOT / "models" / "realworld_v2" / "upstream" / "ViT-L-14.pt"
UPSTREAM_HEAD_PATH = (
    PROJECT_ROOT
    / "third_party"
    / "UniversalFakeDetect"
    / "pretrained_weights"
    / "fc_weights.pth"
)
DEFAULT_TRAINED_HEAD_PATH = PROJECT_ROOT / "models" / "realworld_v2" / "realworld_v2_best.pth"
FEATURE_DIM = 768
EXPECTED_BACKBONE_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_vendor_module() -> ModuleType:
    if not VENDOR_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "The vendored UniversalFakeDetect CLIP implementation is missing. "
            "See third_party/UniversalFakeDetect."
        )
    module_name = "_realworld_v2_vendor_clip_model"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, VENDOR_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import CLIP implementation from {VENDOR_MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_clip_backbone(
    path: Path = DEFAULT_BACKBONE_PATH,
    *,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load a checksum-verifiable OpenAI CLIP JIT archive without tokenizer deps."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing CLIP backbone: {path}. Run scripts/download_realworld_assets.ps1."
        )
    if path.resolve() == DEFAULT_BACKBONE_PATH.resolve():
        actual_hash = _sha256_file(path)
        if actual_hash != EXPECTED_BACKBONE_SHA256:
            raise RuntimeError(
                "The cached CLIP backbone failed SHA-256 verification. "
                "Do not use it; download the official asset again."
            )
    module = _load_vendor_module()
    with path.open("rb") as stream:
        archive = torch.jit.load(stream, map_location="cpu").eval()
    backbone = module.build_model(archive.state_dict()).to(device)
    if torch.device(device).type == "cpu":
        backbone.float()
    backbone.eval()
    backbone.requires_grad_(False)
    return backbone


def _extract_head_state(checkpoint: object) -> tuple[Mapping[str, Tensor], dict[str, Any]]:
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Binary-head checkpoint is not a mapping.")
    if "head_state_dict" in checkpoint:
        state = checkpoint["head_state_dict"]
        metadata = dict(checkpoint.get("metadata", {}))
    else:
        state = checkpoint
        metadata = {}
    if not isinstance(state, Mapping) or set(state) != {"weight", "bias"}:
        raise RuntimeError("Binary-head checkpoint must contain weight and bias tensors.")
    return state, metadata  # type: ignore[return-value]


class CLIPBinaryClassifier(nn.Module):
    """Frozen CLIP visual features followed by a trainable real/AI logit."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(FEATURE_DIM, 1)

    def forward(self, images: Tensor) -> Tensor:
        features = self.backbone.encode_image(images)
        return self.head(features.float()).flatten()


def load_binary_classifier(
    *,
    device: torch.device | str,
    backbone_path: Path = DEFAULT_BACKBONE_PATH,
    head_path: Path | None = None,
    initialize_head: str = "best_available",
) -> tuple[CLIPBinaryClassifier, dict[str, Any]]:
    """Load the v2 head, academic upstream head, or a new random head."""

    resolved_device = torch.device(device)
    backbone = load_clip_backbone(backbone_path, device=resolved_device)
    model = CLIPBinaryClassifier(backbone).to(resolved_device)
    metadata: dict[str, Any] = {
        "model_id": "realworld_v2",
        "backbone": "OpenAI CLIP ViT-L/14",
        "threshold": 0.5,
    }

    selected_head: Path | None = head_path
    if selected_head is None and initialize_head == "best_available":
        selected_head = (
            DEFAULT_TRAINED_HEAD_PATH
            if DEFAULT_TRAINED_HEAD_PATH.is_file()
            else UPSTREAM_HEAD_PATH
        )
    elif selected_head is None and initialize_head == "upstream":
        selected_head = UPSTREAM_HEAD_PATH
    elif initialize_head == "random":
        selected_head = None
    elif selected_head is None:
        raise ValueError("initialize_head must be best_available, upstream, or random")

    if selected_head is not None:
        if not selected_head.is_file():
            raise FileNotFoundError(f"Missing binary-head checkpoint: {selected_head}")
        checkpoint = torch.load(selected_head, map_location="cpu", weights_only=True)
        state, saved_metadata = _extract_head_state(checkpoint)
        model.head.load_state_dict(state, strict=True)
        metadata.update(saved_metadata)
        metadata["head_path"] = str(selected_head)
        if selected_head.resolve() == UPSTREAM_HEAD_PATH.resolve():
            metadata.update(
                {
                    "model_id": "universal_fake_detect_academic_reference",
                    "usage_track": "academic_only",
                    "threshold": 0.5,
                }
            )
    else:
        metadata.update({"head_path": None, "usage_track": "untrained"})

    model.eval()
    return model, metadata


__all__ = [
    "CLIPBinaryClassifier",
    "DEFAULT_BACKBONE_PATH",
    "DEFAULT_TRAINED_HEAD_PATH",
    "EXPECTED_BACKBONE_SHA256",
    "FEATURE_DIM",
    "UPSTREAM_HEAD_PATH",
    "load_binary_classifier",
    "load_clip_backbone",
]

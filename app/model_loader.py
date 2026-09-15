"""Construct and load the four trained AI-image detector models."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import TypedDict

import torch
from torch import Tensor, nn
from torchvision import models

try:
    import timm
except ImportError:  # Keep the error actionable if only torchvision is present.
    timm = None  # type: ignore[assignment]


AI_CLASS_INDEX = 0
REAL_CLASS_INDEX = 1
CLASS_LABELS = {
    AI_CLASS_INDEX: "AI-generated",
    REAL_CLASS_INDEX: "Real",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "tuned"


class ModelSpec(TypedDict):
    """Serializable metadata needed to construct and locate a model."""

    display_name: str
    architecture: str
    checkpoint_filename: str
    num_classes: int
    ai_class_index: int
    real_class_index: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "resnet50": {
        "display_name": "ResNet50",
        "architecture": "torchvision.models.resnet50",
        "checkpoint_filename": "resnet50_hyperparameter_tuned_best.pth",
        "num_classes": 2,
        "ai_class_index": AI_CLASS_INDEX,
        "real_class_index": REAL_CLASS_INDEX,
    },
    "efficientnetv2s": {
        "display_name": "EfficientNetV2-S",
        "architecture": "torchvision.models.efficientnet_v2_s",
        "checkpoint_filename": "efficientnetv2s_hyperparameter_tuned_best.pth",
        "num_classes": 2,
        "ai_class_index": AI_CLASS_INDEX,
        "real_class_index": REAL_CLASS_INDEX,
    },
    "vit16": {
        "display_name": "ViT-B/16",
        "architecture": "torchvision.models.vit_b_16",
        "checkpoint_filename": "vit16_hyperparameter_tuned_best.pth",
        "num_classes": 2,
        "ai_class_index": AI_CLASS_INDEX,
        "real_class_index": REAL_CLASS_INDEX,
    },
    "xception": {
        "display_name": "Xception",
        "architecture": "timm.models.legacy_xception",
        "checkpoint_filename": "xception_hyperparameter_tuned_best.pth",
        "num_classes": 2,
        "ai_class_index": AI_CLASS_INDEX,
        "real_class_index": REAL_CLASS_INDEX,
    },
}

_MODEL_ALIASES = {
    "efficientnet_v2_s": "efficientnetv2s",
    "efficientnet-v2-s": "efficientnetv2s",
    "efficientnetv2-s": "efficientnetv2s",
    "vit_b_16": "vit16",
    "vit-b/16": "vit16",
    "legacy_xception": "xception",
}


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve an inference device, preferring CUDA when one is available."""

    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid PyTorch device {device!r}.") from exc

    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but PyTorch cannot access a CUDA device. "
                "Use device='cpu' or install a CUDA-enabled PyTorch build."
            )
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {resolved.index} does not exist; "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )

    return resolved


def _canonical_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower()
    canonical_name = _MODEL_ALIASES.get(normalized, normalized)
    if canonical_name not in MODEL_SPECS:
        supported = ", ".join(MODEL_SPECS)
        raise KeyError(f"Unknown model {model_name!r}. Supported models: {supported}.")
    return canonical_name


def build_model(model_name: str) -> nn.Module:
    """Build one uninitialized two-class architecture without downloading weights."""

    model_name = _canonical_model_name(model_name)

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    if model_name == "efficientnetv2s":
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        return model

    if model_name == "vit16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
        return model

    if timm is None:
        raise ImportError(
            "Loading Xception requires the 'timm' package. Install the app "
            "requirements and try again."
        )
    return timm.create_model("legacy_xception", pretrained=False, num_classes=2)


def _checkpoint_path(
    model_name: str,
    project_root: str | PathLike[str] | None,
) -> Path:
    root = (
        PROJECT_ROOT
        if project_root is None
        else Path(project_root).expanduser().resolve()
    )
    return root / "models" / "tuned" / MODEL_SPECS[model_name]["checkpoint_filename"]


def _extract_state_dict(checkpoint: object, checkpoint_path: Path) -> Mapping[str, Tensor]:
    # The current tuned checkpoints are direct state dictionaries.  Supporting
    # the common {"state_dict": ...} wrapper keeps errors clear if artifacts are
    # exported that way in the future while retaining weights-only deserialization.
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if not isinstance(checkpoint, Mapping) or not all(
        isinstance(key, str) for key in checkpoint
    ):
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' does not contain a valid model state dict."
        )
    return checkpoint  # type: ignore[return-value]


def _load_model(
    model_name: str,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    spec = MODEL_SPECS[model_name]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint for {spec['display_name']}: '{checkpoint_path}'. "
            "Expected the trained .pth file under models/tuned; this application "
            "does not train or download models."
        )

    model = build_model(model_name).to(device)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not safely load the {spec['display_name']} checkpoint "
            f"'{checkpoint_path}': {exc}"
        ) from exc

    state_dict = _extract_state_dict(checkpoint, checkpoint_path)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' is incompatible with the configured "
            f"{spec['display_name']} architecture. Check the checkpoint and installed "
            "torchvision/timm versions."
        ) from exc

    model.eval()
    model.requires_grad_(False)
    return model


def load_models(
    project_root: str | PathLike[str] | None = None,
    device: str | torch.device | None = None,
) -> tuple[dict[str, nn.Module], torch.device]:
    """Load all tuned checkpoints and return ``(models_by_name, device)``.

    The returned keys preserve ``MODEL_SPECS`` order.  Call this function once
    when creating the inference service and reuse the returned model instances.
    """

    resolved_device = resolve_device(device)
    loaded_models: dict[str, nn.Module] = {}

    for model_name in MODEL_SPECS:
        checkpoint_path = _checkpoint_path(model_name, project_root)
        loaded_models[model_name] = _load_model(
            model_name,
            checkpoint_path,
            resolved_device,
        )

    return loaded_models, resolved_device


__all__ = [
    "AI_CLASS_INDEX",
    "REAL_CLASS_INDEX",
    "CLASS_LABELS",
    "PROJECT_ROOT",
    "DEFAULT_MODEL_DIR",
    "MODEL_SPECS",
    "ModelSpec",
    "resolve_device",
    "build_model",
    "load_models",
]

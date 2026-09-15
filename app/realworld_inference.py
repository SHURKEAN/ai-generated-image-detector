"""Single-image inference with the cross-generator CLIP detector."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps

from realworld_v2.dataset import evaluation_transform
from realworld_v2.model import (
    DEFAULT_BACKBONE_PATH,
    DEFAULT_TRAINED_HEAD_PATH,
    UPSTREAM_HEAD_PATH,
    load_binary_classifier,
)


def realworld_assets_available() -> bool:
    return DEFAULT_BACKBONE_PATH.is_file() and (
        DEFAULT_TRAINED_HEAD_PATH.is_file() or UPSTREAM_HEAD_PATH.is_file()
    )


class RealWorldImageDetector:
    def __init__(
        self,
        device: str | torch.device | None = None,
        *,
        initialize_head: str = "best_available",
    ) -> None:
        self.requested_device = device
        self.initialize_head = initialize_head
        self.device: torch.device | None = None
        self.model: torch.nn.Module | None = None
        self.metadata: dict[str, Any] = {}
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def load(self) -> None:
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            if self.requested_device is None or str(self.requested_device).lower() == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(self.requested_device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")
            model, metadata = load_binary_classifier(
                device=device,
                initialize_head=self.initialize_head,
            )
            self.model = model
            self.metadata = metadata
            self.device = device

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.model is not None,
            "device": str(self.device) if self.device else "not loaded",
            "model_id": self.metadata.get("model_id", "realworld_v2"),
        }

    def predict(self, image: Image.Image) -> dict[str, Any]:
        if image is None or not isinstance(image, Image.Image):
            raise ValueError("Please upload a valid image.")
        self.load()
        assert self.model is not None and self.device is not None
        processed = ImageOps.exif_transpose(image).convert("RGB")
        tensor = evaluation_transform()(processed).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with self._inference_lock, torch.inference_mode():
            ai_probability = float(torch.sigmoid(self.model(tensor).float())[0].cpu())
        threshold = float(self.metadata.get("threshold", 0.5))
        is_ai = ai_probability >= threshold
        locally_trained = self.metadata.get("model_id") == "realworld_v2"
        model_id = str(self.metadata.get("model_id", "realworld_v2"))
        return {
            "label": "AI-generated" if is_ai else "Real",
            "confidence_percent": (ai_probability if is_ai else 1.0 - ai_probability) * 100.0,
            "ensemble_ai_probability": ai_probability,
            "individual_ai_probabilities": {"CLIP ViT-L/14 detector": ai_probability},
            "detector_version": model_id,
            "reliability_note": (
                "Locally trained real-world v2 head; this remains a probabilistic screening result."
                if locally_trained
                else "Academic cross-generator reference model. Use this as screening evidence, not forensic proof."
            ),
            "threshold": threshold,
            "device": str(self.device),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }


__all__ = ["RealWorldImageDetector", "realworld_assets_available"]

"""Single-image inference using the CVPR 2025 Community Forensics detector."""

from __future__ import annotations

import threading
import time
from typing import Any

import torch
from PIL import Image, ImageOps

from realworld_v2.community_model import community_transform, load_community_model


class CommunityForensicsDetector:
    def __init__(self, device: str | torch.device | None = None) -> None:
        self.requested_device = device
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
            self.model, self.metadata = load_community_model(device)
            self.device = device

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.model is not None,
            "device": str(self.device) if self.device else "not loaded",
            "model_id": self.metadata.get("model_id", "community_forensics_cvpr2025"),
        }

    def predict(self, image: Image.Image) -> dict[str, Any]:
        if image is None or not isinstance(image, Image.Image):
            raise ValueError("Please upload a valid image.")
        self.load()
        assert self.model is not None and self.device is not None
        processed = ImageOps.exif_transpose(image).convert("RGB")
        tensor = community_transform()(processed).unsqueeze(0).to(self.device)
        started = time.perf_counter()
        with self._inference_lock, torch.inference_mode():
            logit = self.model(tensor).flatten()[0]
            ai_probability = float(torch.sigmoid(logit.float()).cpu())
        threshold = float(self.metadata.get("threshold", 0.5))
        is_ai = ai_probability >= threshold
        return {
            "label": "AI-generated" if is_ai else "Real",
            "confidence_percent": (ai_probability if is_ai else 1.0 - ai_probability) * 100.0,
            "ensemble_ai_probability": ai_probability,
            "individual_ai_probabilities": {"Community Forensics ViT-S/16": ai_probability},
            "detector_version": str(self.metadata["model_id"]),
            "reliability_note": (
                "Cross-generator academic detector trained across thousands of generators. "
                "The result is probabilistic screening, not forensic proof."
            ),
            "threshold": threshold,
            "device": str(self.device),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }


__all__ = ["CommunityForensicsDetector"]


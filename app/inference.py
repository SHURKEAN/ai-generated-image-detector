"""Single-image inference for the tuned AI-image ensemble."""

from __future__ import annotations

import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .ensemble import DEFAULT_ENSEMBLE_WEIGHTS, weighted_soft_vote
from .model_loader import load_models
from .preprocessing import preprocess_image


CLASS_NAMES = {0: "AI-generated", 1: "Real"}


class ImageDetector:
    """Load the four tuned models once and classify uploaded images."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.requested_device = device
        self.models: dict[str, torch.nn.Module] | None = None
        self.device: torch.device | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def load(self) -> None:
        """Load and validate all checkpoints, once per detector instance."""

        if self.models is not None:
            return
        with self._load_lock:
            if self.models is not None:
                return
            models, resolved_device = load_models(
                project_root=self.project_root,
                device=self.requested_device,
            )
            self.models = models
            self.device = resolved_device

    def status(self) -> dict[str, Any]:
        """Return lightweight runtime information for the UI or health checks."""

        return {
            "loaded": self.models is not None,
            "device": str(self.device) if self.device is not None else "not loaded",
            "model_count": len(self.models) if self.models is not None else 0,
        }

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Classify one PIL image and return UI-friendly result values."""

        if image is None:
            raise ValueError("Please upload an image before running the detector.")
        if not isinstance(image, Image.Image):
            raise TypeError("The detector expects a PIL image.")

        self.load()
        assert self.models is not None
        assert self.device is not None

        tensor = preprocess_image(image).unsqueeze(0).to(self.device)
        probability_vectors: dict[str, torch.Tensor] = {}
        started = time.perf_counter()

        # A lock keeps concurrent Gradio requests from competing for the same
        # set of GPU models. This can be relaxed later after load testing.
        with self._inference_lock, torch.inference_mode():
            for name, model in self.models.items():
                logits = model(tensor)
                if logits.shape != (1, 2):
                    raise RuntimeError(
                        f"{name} returned logits with shape {tuple(logits.shape)}; "
                        "expected (1, 2)."
                    )
                probability_vectors[name] = torch.softmax(
                    logits.float(), dim=1
                ).squeeze(0).cpu()

        ensemble_probabilities = weighted_soft_vote(
            probability_vectors,
            DEFAULT_ENSEMBLE_WEIGHTS,
        )
        predicted_index = int(torch.argmax(ensemble_probabilities).item())
        predicted_probability = float(ensemble_probabilities[predicted_index].item())

        return {
            "label": CLASS_NAMES[predicted_index],
            "confidence_percent": predicted_probability * 100.0,
            "ensemble_ai_probability": float(ensemble_probabilities[0].item()),
            "individual_ai_probabilities": {
                name: float(probabilities[0].item())
                for name, probabilities in probability_vectors.items()
            },
            "detector_version": "published_ctds_v1",
            "reliability_note": (
                "Published CIFAKE baseline. Its score is not calibrated for arbitrary "
                "phone photographs or unseen image generators."
            ),
            "threshold": 0.5,
            "device": str(self.device),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }


@lru_cache(maxsize=1)
def get_detector() -> ImageDetector:
    """Return the process-wide detector used by the Gradio application."""

    requested_device = os.getenv("AI_DETECTOR_DEVICE") or None
    return ImageDetector(device=requested_device)


__all__ = [
    "CLASS_NAMES",
    "DEFAULT_ENSEMBLE_WEIGHTS",
    "ImageDetector",
    "get_detector",
    "weighted_soft_vote",
]

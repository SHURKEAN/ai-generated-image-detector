from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from torch import nn

from app.inference import ImageDetector


class FixedLogitsModel(nn.Module):
    def __init__(self, fake_logit: float, real_logit: float) -> None:
        super().__init__()
        self.register_buffer(
            "fixed_logits", torch.tensor([[fake_logit, real_logit]], dtype=torch.float32)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fixed_logits.repeat(inputs.shape[0], 1)


def detector_with_logits(fake_logit: float, real_logit: float) -> ImageDetector:
    detector = ImageDetector(device="cpu")
    detector.device = torch.device("cpu")
    detector.models = {
        "resnet50": FixedLogitsModel(fake_logit, real_logit).eval(),
        "efficientnetv2s": FixedLogitsModel(fake_logit, real_logit).eval(),
        "vit16": FixedLogitsModel(fake_logit, real_logit).eval(),
        "xception": FixedLogitsModel(fake_logit, real_logit).eval(),
    }
    return detector


def test_predict_maps_class_zero_to_ai_generated() -> None:
    detector = detector_with_logits(fake_logit=3.0, real_logit=1.0)
    image = Image.new("RGB", (32, 32), "white")

    with patch(
        "app.inference.preprocess_image",
        return_value=torch.zeros((3, 224, 224), dtype=torch.float32),
    ):
        result = detector.predict(image)

    assert result["label"] == "AI-generated"
    assert result["ensemble_ai_probability"] > 0.5
    assert result["confidence_percent"] > 50.0
    assert set(result["individual_ai_probabilities"]) == {
        "resnet50",
        "efficientnetv2s",
        "vit16",
        "xception",
    }


def test_predict_maps_class_one_to_real() -> None:
    detector = detector_with_logits(fake_logit=-1.0, real_logit=2.0)
    image = Image.new("RGB", (32, 32), "black")

    with patch(
        "app.inference.preprocess_image",
        return_value=torch.zeros((3, 224, 224), dtype=torch.float32),
    ):
        result = detector.predict(image)

    assert result["label"] == "Real"
    assert result["ensemble_ai_probability"] < 0.5
    assert result["confidence_percent"] > 50.0


@pytest.mark.skipif(
    os.getenv("RUN_MODEL_INTEGRATION") != "1",
    reason="Set RUN_MODEL_INTEGRATION=1 to load the real checkpoints.",
)
def test_real_checkpoint_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    detector = ImageDetector(project_root=root)
    for folder, expected_label in (("FAKE", "AI-generated"), ("REAL", "Real")):
        sample_candidates = sorted(
            (root / "CIFAKE_FULL" / "test" / folder).glob("*.jpg")
        )
        assert sample_candidates, f"No CIFAKE {folder} test image is available."

        with Image.open(sample_candidates[0]) as image:
            result = detector.predict(image.copy())

        assert result["label"] == expected_label
        assert 0.0 <= result["ensemble_ai_probability"] <= 1.0
        assert len(result["individual_ai_probabilities"]) == 4

from __future__ import annotations

import torch
from PIL import Image

from app.preprocessing import preprocess_image


def test_preprocessing_accepts_non_rgb_images() -> None:
    image = Image.new("L", (40, 20), 128)

    tensor = preprocess_image(image)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()

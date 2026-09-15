"""Transparent domain routing between the publication and real-world detector."""

from __future__ import annotations

from typing import Any, Protocol

from PIL import Image

from .inference import ImageDetector


class PredictingDetector(Protocol):
    def predict(self, image: Image.Image) -> dict[str, Any]: ...


class AdaptiveImageDetector:
    """Use the published detector only for native CIFAKE-scale inputs.

    This is a declared input-domain route, not an ensemble fitted on test data.
    The original dimensions are checked before either model resizes the image.
    """

    def __init__(
        self,
        realworld_detector: PredictingDetector,
        *,
        published_detector: PredictingDetector | None = None,
        cifake_max_dimension: int = 64,
    ) -> None:
        self.realworld_detector = realworld_detector
        self.published_detector = published_detector or ImageDetector()
        self.cifake_max_dimension = cifake_max_dimension

    def predict(self, image: Image.Image) -> dict[str, Any]:
        if image is None or not isinstance(image, Image.Image):
            raise ValueError("Please upload a valid image.")
        is_cifake_scale = max(image.size) <= self.cifake_max_dimension
        detector = self.published_detector if is_cifake_scale else self.realworld_detector
        result = dict(detector.predict(image))
        route = "published CIFAKE-scale route" if is_cifake_scale else "real-world route"
        result["detector_version"] = f"adaptive_v2 / {result.get('detector_version', 'unknown')}"
        existing_note = str(result.get("reliability_note", "")).strip()
        result["reliability_note"] = f"Selected {route}. {existing_note}".strip()
        result["routing"] = {
            "original_width": image.width,
            "original_height": image.height,
            "cifake_max_dimension": self.cifake_max_dimension,
            "selected_route": route,
        }
        return result


__all__ = ["AdaptiveImageDetector"]


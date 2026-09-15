"""Select the real-world detector while retaining the published baseline."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Protocol

from PIL import Image

from realworld_v2.community_model import community_assets_available

from .adaptive_inference import AdaptiveImageDetector
from .community_inference import CommunityForensicsDetector
from .inference import ImageDetector
from .realworld_inference import RealWorldImageDetector, realworld_assets_available


class Detector(Protocol):
    def predict(self, image: Image.Image) -> dict[str, object]: ...


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    mode = os.getenv("AI_DETECTOR_MODE", "auto").strip().lower()
    device = os.getenv("AI_DETECTOR_DEVICE") or None
    if mode not in {"auto", "realworld", "community", "universal", "published"}:
        raise ValueError(
            "AI_DETECTOR_MODE must be auto, realworld, community, universal, or published"
        )
    if mode == "realworld":
        return RealWorldImageDetector(device=device)
    if mode == "community":
        return CommunityForensicsDetector(device=device)
    if mode == "universal":
        return RealWorldImageDetector(device=device, initialize_head="upstream")
    # The local realworld_v2 checkpoint remains available through the explicit
    # `realworld` mode, but it must not replace a stronger validated detector
    # merely because its file exists.  Community Forensics currently wins the
    # held-out and external acceptance comparisons, so auto mode prefers it.
    if mode == "auto" and community_assets_available():
        return AdaptiveImageDetector(
            CommunityForensicsDetector(device=device),
            published_detector=ImageDetector(device=device),
        )
    if mode == "auto" and realworld_assets_available():
        return AdaptiveImageDetector(
            RealWorldImageDetector(device=device),
            published_detector=ImageDetector(device=device),
        )
    return ImageDetector(device=device)


__all__ = ["get_detector"]

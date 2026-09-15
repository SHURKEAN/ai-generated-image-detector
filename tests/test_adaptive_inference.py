from __future__ import annotations

from PIL import Image

from app.adaptive_inference import AdaptiveImageDetector


class StubDetector:
    def __init__(self, version: str) -> None:
        self.version = version
        self.calls = 0

    def predict(self, image: Image.Image) -> dict[str, object]:
        self.calls += 1
        return {
            "label": "Real",
            "confidence_percent": 80.0,
            "ensemble_ai_probability": 0.2,
            "individual_ai_probabilities": {self.version: 0.2},
            "detector_version": self.version,
            "reliability_note": "stub",
        }


def test_small_images_use_published_route() -> None:
    published = StubDetector("published")
    realworld = StubDetector("realworld")
    detector = AdaptiveImageDetector(realworld, published_detector=published)

    result = detector.predict(Image.new("RGB", (32, 32)))

    assert published.calls == 1
    assert realworld.calls == 0
    assert result["routing"]["selected_route"] == "published CIFAKE-scale route"


def test_phone_sized_images_use_realworld_route() -> None:
    published = StubDetector("published")
    realworld = StubDetector("realworld")
    detector = AdaptiveImageDetector(realworld, published_detector=published)

    result = detector.predict(Image.new("RGB", (1200, 1600)))

    assert published.calls == 0
    assert realworld.calls == 1
    assert result["routing"]["selected_route"] == "real-world route"


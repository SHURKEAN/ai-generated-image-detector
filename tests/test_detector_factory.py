from app import detector_factory
from app.adaptive_inference import AdaptiveImageDetector


class _FakeCommunity:
    def __init__(self, device=None):
        self.device = device


class _FakePublished:
    def __init__(self, device=None):
        self.device = device


class _FakeRealWorld:
    def __init__(self, device=None, *, initialize_head="best_available"):
        self.device = device
        self.initialize_head = initialize_head


def _patch_detectors(monkeypatch):
    monkeypatch.setattr(detector_factory, "CommunityForensicsDetector", _FakeCommunity)
    monkeypatch.setattr(detector_factory, "ImageDetector", _FakePublished)
    monkeypatch.setattr(detector_factory, "RealWorldImageDetector", _FakeRealWorld)
    detector_factory.get_detector.cache_clear()


def test_auto_prefers_validated_community_route(monkeypatch):
    _patch_detectors(monkeypatch)
    monkeypatch.delenv("AI_DETECTOR_MODE", raising=False)
    monkeypatch.setattr(detector_factory, "community_assets_available", lambda: True)
    monkeypatch.setattr(detector_factory, "realworld_assets_available", lambda: True)

    detector = detector_factory.get_detector()

    assert isinstance(detector, AdaptiveImageDetector)
    assert isinstance(detector.realworld_detector, _FakeCommunity)


def test_universal_mode_cannot_silently_load_local_head(monkeypatch):
    _patch_detectors(monkeypatch)
    monkeypatch.setenv("AI_DETECTOR_MODE", "universal")

    detector = detector_factory.get_detector()

    assert isinstance(detector, _FakeRealWorld)
    assert detector.initialize_head == "upstream"


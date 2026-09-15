from __future__ import annotations

from realworld_v2.dataset import source_balanced_sampler
from realworld_v2.manifest import ManifestRecord


def record(label: str, source: str, image_id: str) -> ManifestRecord:
    return ManifestRecord(
        image_id=image_id,
        image_path=f"{image_id}.jpg",
        label=label,
        source_id=source,
        source_kind="test",
        generator_or_camera=source,
        license="test",
        license_url="test",
        usage_track="academic_only",
        consent_status="n/a",
        split="train",
        sha256="0" * 64,
    )


def test_sampler_balances_labels_before_sources() -> None:
    records = [
        record("real", "one-real-source", "real-1"),
        record("real", "one-real-source", "real-2"),
        record("ai", "ai-source-a", "ai-1"),
        record("ai", "ai-source-b", "ai-2"),
        record("ai", "ai-source-c", "ai-3"),
    ]

    sampler = source_balanced_sampler(records)
    real_mass = sum(float(weight) for weight, item in zip(sampler.weights, records) if item.label == "real")
    ai_mass = sum(float(weight) for weight, item in zip(sampler.weights, records) if item.label == "ai")

    assert real_mass == ai_mass


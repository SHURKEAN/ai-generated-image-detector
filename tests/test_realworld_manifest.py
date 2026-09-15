from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from realworld_v2.manifest import ManifestError, ManifestRecord, validate_manifest


def make_record(path: Path, *, source: str, split: str, label: str = "real") -> ManifestRecord:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ManifestRecord(
        image_id=digest[:20],
        image_path=str(path),
        label=label,
        source_id=source,
        source_kind="test",
        generator_or_camera="test-device",
        license="test-license",
        license_url="test-license-record",
        usage_track="product_safe",
        consent_status="test-consent",
        split=split,
        sha256=digest,
        notes="",
    )


def test_source_leakage_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    records = [
        make_record(first, source="same-camera", split="train"),
        make_record(second, source="same-camera", split="test"),
    ]

    with pytest.raises(ManifestError, match="leaks across splits"):
        validate_manifest(records)


def test_duplicate_content_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"identical")
    second.write_bytes(b"identical")
    records = [
        make_record(first, source="source-one", split="train"),
        make_record(second, source="source-two", split="validation"),
    ]

    with pytest.raises(ManifestError, match="duplicate image content"):
        validate_manifest(records)


def test_private_image_cannot_enter_product_training(tmp_path: Path) -> None:
    image = tmp_path / "private.jpg"
    image.write_bytes(b"private")
    original = make_record(image, source="private", split="external_test")
    private = ManifestRecord(**{**original.__dict__, "usage_track": "private_evaluation"})

    with pytest.raises(ManifestError, match="no records are eligible"):
        validate_manifest([private], training_track="product_safe")


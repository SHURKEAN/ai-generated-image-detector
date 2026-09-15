from __future__ import annotations

import json
from pathlib import Path

from realworld_v2.manifest import build_manifest


def test_build_manifest_reads_explicit_registry_include(tmp_path: Path) -> None:
    image = tmp_path / "sample.jpg"
    image.write_bytes(b"image-placeholder")
    child = tmp_path / "child.json"
    child.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "enabled": True,
                        "path": str(image),
                        "label": "real",
                        "source_id": "included-camera",
                        "source_kind": "camera",
                        "generator_or_camera": "test",
                        "license": "test-license",
                        "license_url": "test-record",
                        "usage_track": "product_safe",
                        "consent_status": "consented",
                        "split": "train",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    parent = tmp_path / "parent.json"
    parent.write_text(
        json.dumps({"include_registries": ["child.json"], "sources": []}),
        encoding="utf-8",
    )

    records = build_manifest(parent, tmp_path / "manifest.csv", project_root=tmp_path)

    assert len(records) == 1
    assert records[0].source_id == "included-camera"


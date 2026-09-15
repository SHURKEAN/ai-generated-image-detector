"""Verify that every published CTDS checkpoint still matches its frozen manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "models" / "published_ctds_v1" / "manifest.json"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []

    for item in manifest["checkpoints"]:
        path = PROJECT_ROOT / item["path"]
        if not path.is_file():
            failures.append(f"MISSING  {item['path']}")
            continue
        if path.stat().st_size != int(item["bytes"]):
            failures.append(f"SIZE     {item['path']}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != item["sha256"]:
            failures.append(f"SHA256   {item['path']}")
            continue
        print(f"OK       {item['model']}: {actual_hash}")

    if failures:
        print("\nPublished baseline verification FAILED:")
        print("\n".join(failures))
        return 1

    print(f"\nVerified {len(manifest['checkpoints'])} immutable published checkpoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


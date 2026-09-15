"""Build and validate provenance-first image manifests.

The source registry makes licensing and split decisions explicit before an
image can enter training. A source ID may occur in only one split, preventing
near-duplicate images or one generator/device from leaking across splits.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "data" / "source_registry.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "realworld_v2.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LABELS = {"real", "ai"}
SPLITS = {"train", "validation", "test", "external_test"}
USAGE_TRACKS = {"product_safe", "academic_only", "private_evaluation"}
REQUIRED_FIELDS = (
    "image_id",
    "image_path",
    "label",
    "source_id",
    "source_kind",
    "generator_or_camera",
    "license",
    "license_url",
    "usage_track",
    "consent_status",
    "split",
    "sha256",
    "notes",
)


class ManifestError(ValueError):
    """Raised when provenance, licensing, or split invariants are violated."""


@dataclass(frozen=True)
class ManifestRecord:
    image_id: str
    image_path: str
    label: str
    source_id: str
    source_kind: str
    generator_or_camera: str
    license: str
    license_url: str
    usage_track: str
    consent_status: str
    split: str
    sha256: str
    notes: str = ""

    @classmethod
    def from_mapping(cls, row: dict[str, str]) -> "ManifestRecord":
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ManifestError(f"Manifest row is missing fields: {missing}")
        return cls(**{field: str(row.get(field, "")).strip() for field in REQUIRED_FIELDS})


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path
        return
    if not path.is_dir():
        raise ManifestError(f"Enabled source path does not exist: {path}")
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            yield candidate


def _load_registry_sources(
    registry_path: Path,
    *,
    project_root: Path,
    seen: set[Path] | None = None,
) -> list[dict[str, Any]]:
    """Load one registry plus explicit includes, rejecting include cycles."""

    resolved = registry_path.resolve()
    visited = seen if seen is not None else set()
    if resolved in visited:
        raise ManifestError(f"Registry include cycle or duplicate include: {resolved}")
    visited.add(resolved)
    if not resolved.is_file():
        raise ManifestError(f"Included registry does not exist: {resolved}")
    registry = json.loads(resolved.read_text(encoding="utf-8"))
    sources = registry.get("sources")
    if not isinstance(sources, list):
        raise ManifestError(f"Registry must contain a 'sources' list: {resolved}")
    combined = list(sources)
    includes = registry.get("include_registries", [])
    if not isinstance(includes, list):
        raise ManifestError(f"include_registries must be a list: {resolved}")
    for include in includes:
        include_path = Path(str(include)).expanduser()
        if not include_path.is_absolute():
            # Includes are relative to the including registry, making copied
            # dataset bundles self-contained.
            include_path = resolved.parent / include_path
        combined.extend(
            _load_registry_sources(
                include_path,
                project_root=project_root,
                seen=visited,
            )
        )
    return combined


def build_manifest(
    registry_path: Path = DEFAULT_REGISTRY,
    output_path: Path = DEFAULT_MANIFEST,
    project_root: Path = PROJECT_ROOT,
) -> list[ManifestRecord]:
    """Hash every enabled registry source and write a deterministic CSV manifest."""

    sources = _load_registry_sources(registry_path, project_root=project_root)

    records: list[ManifestRecord] = []
    for source in sources:
        if not isinstance(source, dict) or not source.get("enabled", False):
            continue
        missing = [
            field
            for field in (
                "path",
                "label",
                "source_id",
                "source_kind",
                "generator_or_camera",
                "license",
                "license_url",
                "usage_track",
                "consent_status",
                "split",
            )
            if not str(source.get(field, "")).strip()
        ]
        if missing:
            raise ManifestError(
                f"Enabled source {source.get('source_id', '<unknown>')!r} "
                f"is missing metadata: {missing}"
            )

        source_path = Path(str(source["path"])).expanduser()
        if not source_path.is_absolute():
            source_path = project_root / source_path

        found = 0
        for image_path in _iter_images(source_path):
            digest = sha256_file(image_path)
            records.append(
                ManifestRecord(
                    image_id=digest[:20],
                    image_path=_display_path(image_path, project_root),
                    label=str(source["label"]).strip().lower(),
                    source_id=str(source["source_id"]).strip(),
                    source_kind=str(source["source_kind"]).strip(),
                    generator_or_camera=str(source["generator_or_camera"]).strip(),
                    license=str(source["license"]).strip(),
                    license_url=str(source["license_url"]).strip(),
                    usage_track=str(source["usage_track"]).strip().lower(),
                    consent_status=str(source["consent_status"]).strip().lower(),
                    split=str(source["split"]).strip().lower(),
                    sha256=digest,
                    notes=str(source.get("notes", "")).strip(),
                )
            )
            found += 1
        if found == 0:
            raise ManifestError(f"Enabled source contains no supported images: {source_path}")

    validate_manifest(records, project_root=project_root, verify_hashes=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return records


def read_manifest(path: Path = DEFAULT_MANIFEST) -> list[ManifestRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ManifestError(f"Manifest has no header: {path}")
        missing = [field for field in REQUIRED_FIELDS if field not in reader.fieldnames]
        if missing:
            raise ManifestError(f"Manifest is missing columns: {missing}")
        return [ManifestRecord.from_mapping(row) for row in reader]


def resolve_image_path(record: ManifestRecord, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(record.image_path).expanduser()
    return path if path.is_absolute() else project_root / path


def validate_manifest(
    records: Sequence[ManifestRecord],
    *,
    project_root: Path = PROJECT_ROOT,
    verify_hashes: bool = False,
    training_track: str | None = None,
    require_training_splits: bool = False,
) -> dict[str, Any]:
    """Validate files, licenses, consent metadata, duplicates, and source splits."""

    if not records:
        raise ManifestError("The manifest contains no images.")
    if training_track not in {None, "product_safe", "academic"}:
        raise ManifestError("training_track must be 'product_safe', 'academic', or omitted.")

    errors: list[str] = []
    hashes: dict[str, list[ManifestRecord]] = defaultdict(list)
    sources_to_splits: dict[str, set[str]] = defaultdict(set)
    ids: Counter[str] = Counter()
    paths: Counter[str] = Counter()

    selected: list[ManifestRecord] = []
    for row_number, record in enumerate(records, start=2):
        ids[record.image_id] += 1
        paths[record.image_path.casefold()] += 1
        hashes[record.sha256.lower()].append(record)
        sources_to_splits[record.source_id].add(record.split)

        if record.label not in LABELS:
            errors.append(f"row {row_number}: invalid label {record.label!r}")
        if record.split not in SPLITS:
            errors.append(f"row {row_number}: invalid split {record.split!r}")
        if record.usage_track not in USAGE_TRACKS:
            errors.append(f"row {row_number}: invalid usage_track {record.usage_track!r}")
        if not record.source_id:
            errors.append(f"row {row_number}: source_id is required")
        if not record.license or record.license.lower() in {"unknown", "unspecified"}:
            errors.append(f"row {row_number}: a verified license is required")
        if not record.license_url:
            errors.append(f"row {row_number}: license_url or provenance document is required")
        if len(record.sha256) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in record.sha256):
            errors.append(f"row {row_number}: invalid SHA-256")

        image_path = resolve_image_path(record, project_root)
        if not image_path.is_file():
            errors.append(f"row {row_number}: missing image {record.image_path!r}")
        elif verify_hashes and sha256_file(image_path) != record.sha256.lower():
            errors.append(f"row {row_number}: hash mismatch for {record.image_path!r}")

        allowed = (
            training_track is None
            or (training_track == "product_safe" and record.usage_track == "product_safe")
            or (
                training_track == "academic"
                and record.usage_track in {"product_safe", "academic_only"}
            )
        )
        if allowed:
            selected.append(record)

    for image_id, count in ids.items():
        if count > 1:
            errors.append(f"duplicate image_id {image_id!r} occurs {count} times")
    for image_path, count in paths.items():
        if count > 1:
            errors.append(f"duplicate image_path {image_path!r} occurs {count} times")
    for digest, matching in hashes.items():
        if len(matching) > 1:
            locations = ", ".join(f"{r.source_id}/{r.split}" for r in matching)
            errors.append(f"duplicate image content {digest[:12]}... occurs in {locations}")
    for source_id, source_splits in sources_to_splits.items():
        if len(source_splits) > 1:
            errors.append(
                f"source_id {source_id!r} leaks across splits: {sorted(source_splits)}"
            )

    if training_track is not None and not selected:
        errors.append(f"no records are eligible for the {training_track!r} track")

    if require_training_splits:
        for split in ("train", "validation"):
            labels = {record.label for record in selected if record.split == split}
            if labels != LABELS:
                errors.append(
                    f"{split!r} must contain both labels for the selected track; got {sorted(labels)}"
                )

    if errors:
        preview = "\n- ".join(errors[:30])
        suffix = f"\n... and {len(errors) - 30} more" if len(errors) > 30 else ""
        raise ManifestError(f"Manifest validation failed:\n- {preview}{suffix}")

    return {
        "records": len(records),
        "selected_records": len(selected),
        "labels": dict(Counter(record.label for record in selected)),
        "splits": dict(Counter(record.split for record in selected)),
        "sources": len({record.source_id for record in selected}),
        "verified_hashes": verify_hashes,
        "training_track": training_track,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build CSV from source_registry.json")
    build.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    build.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)

    validate = subparsers.add_parser("validate", help="validate an existing CSV")
    validate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    validate.add_argument("--verify-hashes", action="store_true")
    validate.add_argument("--track", choices=("product_safe", "academic"), default=None)
    validate.add_argument("--require-training-splits", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "build":
        records = build_manifest(args.registry, args.output)
        summary = validate_manifest(records)
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Wrote {args.output}")
        return 0

    records = read_manifest(args.manifest)
    summary = validate_manifest(
        records,
        verify_hashes=args.verify_hashes,
        training_track=args.track,
        require_training_splits=args.require_training_splits,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

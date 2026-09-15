"""Download and extract a compact, source-held-out academic image subset.

Only four selected Community Forensics-Small shards are cached instead of the
full 278 GB release. Images are selected deterministically, NSFW-flagged rows
are excluded, and a generated source registry plus per-image provenance ledger
are written for the project's normal manifest builder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "OwensLab/CommunityForensics-Small"
DATASET_URL = "https://huggingface.co/datasets/OwensLab/CommunityForensics-Small"
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "academic_only" / "community_forensics_small"
CACHE_ROOT = PROJECT_ROOT / "data" / "cache" / "community_shards"
REGISTRY_PATH = PROJECT_ROOT / "data" / "registries" / "community_forensics_small.json"
LEDGER_PATH = PROJECT_ROOT / "data" / "metadata" / "community_forensics_small.csv"


@dataclass(frozen=True)
class ShardPlan:
    shard: int
    label: str
    split: str
    allowed_sources: frozenset[str] | None
    maximum_per_source: int


def plans(args: argparse.Namespace) -> list[ShardPlan]:
    return [
        # Systematic latent-diffusion models: many independent generators.
        ShardPlan(0, "ai", "train", None, args.train_ai_per_generator),
        # LFM is held out completely for validation.
        ShardPlan(78, "ai", "validation", frozenset({"LFM"}), args.validation_per_source),
        # Genuine VISION camera images are held out completely for validation.
        ShardPlan(116, "real", "validation", frozenset({"VISION"}), args.validation_per_source),
        # Genuine COCO photographs are used only for training.
        ShardPlan(117, "real", "train", frozenset({"COCO"}), args.train_real_count),
    ]


def safe_slug(value: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.").lower() or "source"
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{stem[:70]}-{suffix}"


def selection_key(source: str, original_name: str, row_index: int) -> str:
    return hashlib.sha256(f"{source}\0{original_name}\0{row_index}".encode("utf-8")).hexdigest()


def choose_indices(parquet_path: Path, plan: ShardPlan) -> set[int]:
    columns = ["image_name", "model_name", "label", "nsfw_flag"]
    data = pq.read_table(parquet_path, columns=columns).to_pydict()
    expected_label = 1 if plan.label == "ai" else 0
    candidates: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for index, (name, source, label, nsfw) in enumerate(
        zip(data["image_name"], data["model_name"], data["label"], data["nsfw_flag"])
    ):
        source = str(source)
        if int(label) != expected_label or bool(nsfw):
            continue
        if plan.allowed_sources is not None and source not in plan.allowed_sources:
            continue
        candidates[source].append((selection_key(source, str(name), index), index))

    selected: set[int] = set()
    for source_candidates in candidates.values():
        source_candidates.sort()
        selected.update(index for _, index in source_candidates[: plan.maximum_per_source])
    return selected


def image_extension(format_name: str, original_name: str) -> str:
    formats = {"JPEG": ".jpg", "JPG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
    normalized = str(format_name).upper()
    if normalized in formats:
        return formats[normalized]
    suffix = Path(str(original_name)).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else ".img"


def download_shard(shard: int) -> Path:
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"data/HFCF_small_{shard}.parquet",
            local_dir=CACHE_ROOT,
        )
    )


def extract_plan(
    parquet_path: Path,
    plan: ShardPlan,
    *,
    seen_hashes: set[str],
) -> tuple[list[dict[str, str]], Counter[str]]:
    selected = choose_indices(parquet_path, plan)
    ledger: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    columns = [
        "image_name",
        "format",
        "resolution",
        "mode",
        "image_data",
        "model_name",
        "nsfw_flag",
        "prompt",
        "real_source",
        "subset",
        "split",
        "label",
        "architecture",
    ]
    offset = 0
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=32, columns=columns):
        rows = batch.to_pylist()
        for local_index, row in enumerate(rows):
            row_index = offset + local_index
            if row_index not in selected:
                continue
            image_bytes = bytes(row["image_data"])
            digest = hashlib.sha256(image_bytes).hexdigest()
            if digest in seen_hashes:
                continue
            try:
                with Image.open(BytesIO(image_bytes)) as image:
                    image.verify()
            except (OSError, SyntaxError, ValueError):
                continue

            source = str(row["model_name"])
            source_id = f"community_forensics_small::{plan.label}::{source}"
            source_directory = RAW_ROOT / plan.split / plan.label / safe_slug(source)
            source_directory.mkdir(parents=True, exist_ok=True)
            extension = image_extension(str(row["format"]), str(row["image_name"]))
            destination = source_directory / f"{digest}{extension}"
            if not destination.is_file():
                destination.write_bytes(image_bytes)

            prompt_hash = hashlib.sha256(str(row.get("prompt", "")).encode("utf-8")).hexdigest()
            ledger.append(
                {
                    "image_path": destination.relative_to(PROJECT_ROOT).as_posix(),
                    "sha256": digest,
                    "source_id": source_id,
                    "label": plan.label,
                    "split": plan.split,
                    "dataset": REPO_ID,
                    "shard": str(plan.shard),
                    "row_index": str(row_index),
                    "original_name": str(row["image_name"]),
                    "format": str(row["format"]),
                    "resolution": "x".join(str(value) for value in row["resolution"]),
                    "generator_or_real_source": source,
                    "architecture": str(row["architecture"]),
                    "paired_real_source": str(row["real_source"]),
                    "dataset_subset": str(row["subset"]),
                    "dataset_split": str(row["split"]),
                    "prompt_sha256": prompt_hash,
                    "nsfw_flag": str(bool(row["nsfw_flag"])).lower(),
                }
            )
            counts[source_id] += 1
            seen_hashes.add(digest)
        offset += len(rows)
    return ledger, counts


def write_outputs(ledger: list[dict[str, str]]) -> None:
    if not ledger:
        raise RuntimeError("No images were extracted.")
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(sorted(ledger, key=lambda row: row["image_path"]))

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ledger:
        grouped[row["source_id"]].append(row)
    registry_sources: list[dict[str, Any]] = []
    for source_id, rows in sorted(grouped.items()):
        first = rows[0]
        source_path = str(Path(first["image_path"]).parent.as_posix())
        registry_sources.append(
            {
                "enabled": True,
                "path": source_path,
                "label": first["label"],
                "source_id": source_id,
                "source_kind": "ai_generator" if first["label"] == "ai" else "photograph_collection",
                "generator_or_camera": first["generator_or_real_source"],
                "license": "CC-BY-4.0 dataset release; research-purpose and per-source terms apply",
                "license_url": DATASET_URL,
                "usage_track": "academic_only",
                "consent_status": "not_applicable_dataset_record",
                "split": first["split"],
                "notes": (
                    f"Selected from {REPO_ID}; per-image provenance is in "
                    f"{LEDGER_PATH.relative_to(PROJECT_ROOT).as_posix()}."
                ),
            }
        )
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_by": "scripts/prepare_community_subset.py",
                "dataset": REPO_ID,
                "usage_track": "academic_only",
                "sources": registry_sources,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-ai-per-generator", type=int, default=120)
    parser.add_argument("--train-real-count", type=int, default=2992)
    parser.add_argument("--validation-per-source", type=int, default=1153)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seen_hashes: set[str] = set()
    combined_ledger: list[dict[str, str]] = []
    total_counts: Counter[str] = Counter()
    for plan in plans(args):
        print(f"Preparing shard {plan.shard} ({plan.label}/{plan.split})...", flush=True)
        parquet_path = download_shard(plan.shard)
        rows, counts = extract_plan(parquet_path, plan, seen_hashes=seen_hashes)
        combined_ledger.extend(rows)
        total_counts.update(counts)
        print(f"Extracted {len(rows)} images from shard {plan.shard}.", flush=True)
    write_outputs(combined_ledger)
    summary = {
        "images": len(combined_ledger),
        "sources": len(total_counts),
        "by_label_split": dict(
            Counter(f"{row['label']}/{row['split']}" for row in combined_ledger)
        ),
        "registry": str(REGISTRY_PATH),
        "ledger": str(LEDGER_PATH),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


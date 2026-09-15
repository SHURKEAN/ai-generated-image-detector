"""Quickly check that the v2 detector separates local real and AI folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageOps

from realworld_v2.community_model import community_transform, load_community_model
from realworld_v2.dataset import evaluation_transform
from realworld_v2.metrics import binary_metrics
from realworld_v2.model import load_binary_classifier


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_paths(directory: Path, maximum: int) -> list[Path]:
    paths = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    return paths[:maximum]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--ai-dir", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--detector", choices=("community", "universal"), default="community")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    real_paths = image_paths(args.real_dir, args.max_per_class)
    ai_paths = image_paths(args.ai_dir, args.max_per_class)
    paths = real_paths + ai_paths
    labels = [0.0] * len(real_paths) + [1.0] * len(ai_paths)
    if not real_paths or not ai_paths:
        raise ValueError("Both folders must contain supported image files.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.detector == "community":
        model, metadata = load_community_model(device)
        transform = community_transform()
    else:
        model, metadata = load_binary_classifier(device=device, initialize_head="best_available")
        transform = evaluation_transform()
    probabilities: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(paths), args.batch_size):
            tensors = []
            for path in paths[start : start + args.batch_size]:
                with Image.open(path) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    tensors.append(transform(image))
            batch = torch.stack(tensors).to(device)
            probabilities.extend(torch.sigmoid(model(batch).float()).flatten().cpu().tolist())

    report = {
        "model": metadata,
        "real_directory": str(args.real_dir),
        "ai_directory": str(args.ai_dir),
        "metrics": binary_metrics(labels, probabilities, float(metadata.get("threshold", 0.5))),
        "mean_ai_probability_for_real": sum(probabilities[: labels.count(0.0)]) / labels.count(0.0),
        "mean_ai_probability_for_ai": sum(probabilities[labels.count(0.0) :]) / labels.count(1.0),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

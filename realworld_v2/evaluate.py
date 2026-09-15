"""Evaluate real-world v2 overall and by completely held-out source."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from .community_model import community_assets_available, community_transform, load_community_model
from .dataset import ManifestImageDataset, evaluation_transform, records_for_track
from .manifest import DEFAULT_MANIFEST, read_manifest, validate_manifest
from .metrics import binary_metrics
from .model import DEFAULT_TRAINED_HEAD_PATH, load_binary_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "realworld_v2" / "evaluation.json"


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    records = read_manifest(args.manifest)
    manifest_summary = validate_manifest(records, verify_hashes=args.verify_hashes)
    selected = records_for_track(records, split=args.split, track="evaluation")
    detector_name = args.detector
    if detector_name == "auto":
        detector_name = (
            "trained"
            if DEFAULT_TRAINED_HEAD_PATH.is_file()
            else "community"
            if community_assets_available()
            else "universal"
        )
    transform = community_transform() if detector_name == "community" else evaluation_transform()
    dataset = ManifestImageDataset(selected, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    if detector_name == "community":
        if args.checkpoint is not None:
            raise ValueError("--checkpoint is only valid for trained/universal detectors")
        model, metadata = load_community_model(device)
    else:
        head_path = args.checkpoint if args.checkpoint is not None else None
        initialization = "best_available" if detector_name == "trained" else "upstream"
        model, metadata = load_binary_classifier(
            device=device,
            head_path=head_path,
            initialize_head=initialization,
        )
    threshold = float(metadata.get("threshold", 0.5))
    labels: list[float] = []
    probabilities: list[float] = []
    sources: list[str] = []
    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, batch_labels, batch_sources, image_ids in loader:
            scores = torch.sigmoid(model(images.to(device)).float()).flatten().cpu().tolist()
            labels.extend(batch_labels.tolist())
            probabilities.extend(scores)
            sources.extend(batch_sources)
            predictions.extend(
                {
                    "image_id": image_id,
                    "source_id": source,
                    "true_label": "ai" if label >= 0.5 else "real",
                    "ai_probability": probability,
                    "predicted_label": "ai" if probability >= threshold else "real",
                }
                for image_id, source, label, probability in zip(
                    image_ids, batch_sources, batch_labels.tolist(), scores
                )
            )

    grouped: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for label, probability, source in zip(labels, probabilities, sources):
        grouped[source][0].append(label)
        grouped[source][1].append(probability)

    report = {
        "model": metadata,
        "detector": detector_name,
        "split": args.split,
        "manifest": manifest_summary,
        "overall": binary_metrics(labels, probabilities, threshold),
        "by_source": {
            source: binary_metrics(group_labels, group_probabilities, threshold)
            for source, (group_labels, group_probabilities) in sorted(grouped.items())
        },
        "predictions": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--detector",
        choices=("auto", "community", "trained", "universal"),
        default="auto",
    )
    parser.add_argument("--split", choices=("validation", "test", "external_test"), default="test")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verify-hashes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = evaluate(build_parser().parse_args(argv))
    print(json.dumps({key: report[key] for key in ("model", "split", "overall")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

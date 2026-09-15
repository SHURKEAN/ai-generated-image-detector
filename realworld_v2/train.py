"""Train a source-balanced CLIP binary head without changing the published model."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import (
    ManifestImageDataset,
    evaluation_transform,
    records_for_track,
    source_balanced_sampler,
    training_transform,
)
from .manifest import DEFAULT_MANIFEST, PROJECT_ROOT, read_manifest, validate_manifest
from .metrics import binary_metrics, select_threshold
from .model import DEFAULT_TRAINED_HEAD_PATH, load_binary_classifier


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _data_worker_init(_: int) -> None:
    # Avoid every Windows worker creating a full OpenMP thread pool.
    torch.set_num_threads(1)


def _predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[float], list[float], list[str]]:
    labels: list[float] = []
    probabilities: list[float] = []
    sources: list[str] = []
    model.eval()
    with torch.inference_mode():
        for images, batch_labels, batch_sources, _ in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probabilities.extend(torch.sigmoid(logits.float()).cpu().tolist())
            labels.extend(batch_labels.tolist())
            sources.extend(batch_sources)
    return labels, probabilities, sources


def _source_metrics(
    labels: Sequence[float],
    probabilities: Sequence[float],
    sources: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    grouped: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for label, probability, source in zip(labels, probabilities, sources):
        grouped[source][0].append(label)
        grouped[source][1].append(probability)
    return {
        source: binary_metrics(group_labels, group_probabilities, threshold)
        for source, (group_labels, group_probabilities) in sorted(grouped.items())
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    records = read_manifest(args.manifest)
    manifest_summary = validate_manifest(
        records,
        verify_hashes=args.verify_hashes,
        training_track=args.track,
        require_training_splits=True,
    )
    train_records = records_for_track(records, split="train", track=args.track)
    validation_records = records_for_track(records, split="validation", track=args.track)

    train_dataset = ManifestImageDataset(train_records, transform=training_transform())
    validation_dataset = ManifestImageDataset(validation_records, transform=evaluation_transform())
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=source_balanced_sampler(train_records, args.seed),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=_data_worker_init if args.workers > 0 else None,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        worker_init_fn=_data_worker_init if args.workers > 0 else None,
    )

    if args.track == "product_safe" and args.initialize_head == "upstream":
        raise ValueError(
            "A product-safe model cannot initialize from the research-trained upstream head. "
            "Use --initialize-head random."
        )
    model, initial_metadata = load_binary_classifier(
        device=device,
        initialize_head=args.initialize_head,
    )
    model.backbone.eval()
    model.backbone.requires_grad_(False)
    model.head.train()

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    best_score = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.backbone.eval()
        model.head.train()
        losses: list[float] = []
        for batch_index, (images, labels, _, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if batch_index % 50 == 0 or batch_index == len(train_loader):
                print(
                    f"epoch={epoch} batch={batch_index}/{len(train_loader)} "
                    f"loss={losses[-1]:.5f}",
                    flush=True,
                )

        labels, probabilities, sources = _predict(model, validation_loader, device)
        threshold, validation_metrics = select_threshold(
            labels,
            probabilities,
            maximum_real_false_positive_rate=args.maximum_real_false_positive_rate,
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation": validation_metrics,
            "validation_by_source": _source_metrics(labels, probabilities, sources, threshold),
            "seconds": time.perf_counter() - started,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, sort_keys=True), flush=True)

        score = float(validation_metrics["balanced_accuracy"])
        if score > best_score:
            best_score = score
            metadata = {
                "model_id": "realworld_v2",
                "version": "2.0.0",
                "usage_track": args.track,
                "backbone": "OpenAI CLIP ViT-L/14",
                "class_zero": "real",
                "class_one": "ai",
                "threshold": threshold,
                "maximum_real_false_positive_rate": args.maximum_real_false_positive_rate,
                "manifest": str(args.manifest),
                "manifest_summary": manifest_summary,
                "initial_head": initial_metadata.get("model_id", args.initialize_head),
                "seed": args.seed,
                "epoch": epoch,
                "validation_metrics": validation_metrics,
            }
            checkpoint = {
                "head_state_dict": {
                    key: value.detach().cpu() for key, value in model.head.state_dict().items()
                },
                "metadata": metadata,
            }
            torch.save(checkpoint, args.output)

    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "checkpoint": str(args.output),
        "history": str(history_path),
        "best_balanced_accuracy": best_score,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--track", choices=("product_safe", "academic"), default="product_safe")
    parser.add_argument("--initialize-head", choices=("random", "upstream"), default="random")
    parser.add_argument("--output", type=Path, default=DEFAULT_TRAINED_HEAD_PATH)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--maximum-real-false-positive-rate", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--verify-hashes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

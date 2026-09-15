"""Weighted soft-voting utilities for the tuned CIFAKE ensemble."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


# These are the normalized weights saved by the tuned hybrid experiment.
# ResNet50 is still evaluated and displayed by the app, but its published
# tuned weight is zero and therefore it does not affect the final result.
DEFAULT_ENSEMBLE_WEIGHTS: dict[str, float] = {
    "resnet50": 0.0,
    "efficientnetv2s": 1.0 / 3.0,
    "vit16": 1.0 / 3.0,
    "xception": 1.0 / 3.0,
}


def weighted_soft_vote(
    probabilities: Mapping[str, torch.Tensor | Sequence[float]],
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Combine two-class probability vectors using normalized model weights.

    Probability index 0 is AI-generated/fake and index 1 is real. Model
    outputs must already have been passed through softmax.
    """

    selected_weights = dict(weights or DEFAULT_ENSEMBLE_WEIGHTS)
    if not selected_weights:
        raise ValueError("At least one ensemble weight is required.")

    negative = [name for name, weight in selected_weights.items() if weight < 0]
    if negative:
        raise ValueError(f"Ensemble weights cannot be negative: {negative}")

    positive_names = [
        name for name, weight in selected_weights.items() if float(weight) > 0.0
    ]
    missing = [name for name in positive_names if name not in probabilities]
    if missing:
        raise KeyError(f"Missing probabilities for weighted models: {missing}")

    weight_sum = sum(float(selected_weights[name]) for name in positive_names)
    if weight_sum <= 0.0:
        raise ValueError("The ensemble must contain at least one positive weight.")

    combined = torch.zeros(2, dtype=torch.float64)
    for name in positive_names:
        vector = torch.as_tensor(probabilities[name], dtype=torch.float64).detach().cpu()
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector.squeeze(0)
        if vector.shape != (2,):
            raise ValueError(
                f"{name} must provide exactly two probabilities; got {tuple(vector.shape)}."
            )
        if not torch.isfinite(vector).all():
            raise ValueError(f"{name} returned a non-finite probability.")
        if (vector < 0).any() or (vector > 1).any():
            raise ValueError(f"{name} returned a probability outside [0, 1].")
        if not torch.isclose(vector.sum(), torch.tensor(1.0, dtype=vector.dtype), atol=1e-5):
            raise ValueError(f"{name} probabilities do not sum to 1.")

        normalized_weight = float(selected_weights[name]) / weight_sum
        combined += vector * normalized_weight

    return combined.to(dtype=torch.float32)

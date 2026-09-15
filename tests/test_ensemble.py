from __future__ import annotations

import pytest
import torch

from app.ensemble import DEFAULT_ENSEMBLE_WEIGHTS, weighted_soft_vote


def test_paper_weights_are_normalized() -> None:
    assert DEFAULT_ENSEMBLE_WEIGHTS == {
        "resnet50": 0.0,
        "efficientnetv2s": pytest.approx(1 / 3),
        "vit16": pytest.approx(1 / 3),
        "xception": pytest.approx(1 / 3),
    }
    assert sum(DEFAULT_ENSEMBLE_WEIGHTS.values()) == pytest.approx(1.0)


def test_weighted_soft_vote_uses_both_class_probabilities() -> None:
    probabilities = {
        "resnet50": [0.99, 0.01],
        "efficientnetv2s": [0.60, 0.40],
        "vit16": [0.75, 0.25],
        "xception": [0.30, 0.70],
    }

    result = weighted_soft_vote(probabilities)

    assert torch.allclose(result, torch.tensor([0.55, 0.45]), atol=1e-6)


def test_zero_weight_resnet_does_not_change_result() -> None:
    probabilities = {
        "resnet50": [1.0, 0.0],
        "efficientnetv2s": [0.20, 0.80],
        "vit16": [0.20, 0.80],
        "xception": [0.20, 0.80],
    }
    first = weighted_soft_vote(probabilities)
    probabilities["resnet50"] = [0.0, 1.0]
    second = weighted_soft_vote(probabilities)

    assert torch.equal(first, second)


@pytest.mark.parametrize(
    "invalid",
    ([0.2, 0.2], [1.1, -0.1], [float("nan"), float("nan")]),
)
def test_invalid_probability_vectors_are_rejected(invalid: list[float]) -> None:
    probabilities = {
        "efficientnetv2s": invalid,
        "vit16": [0.5, 0.5],
        "xception": [0.5, 0.5],
    }

    with pytest.raises(ValueError):
        weighted_soft_vote(probabilities)

from __future__ import annotations

from realworld_v2.metrics import binary_metrics, select_threshold


def test_threshold_calibration_respects_real_false_positive_ceiling() -> None:
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.05, 0.20, 0.80, 0.82, 0.90, 0.98]

    threshold, metrics = select_threshold(
        labels,
        scores,
        maximum_real_false_positive_rate=0.0,
    )

    assert threshold > 0.80
    assert metrics["false_positive_rate_real"] == 0.0
    assert metrics["recall_ai"] == 1.0


def test_binary_metrics_use_real_as_zero_and_ai_as_one() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix_real_ai"] == [[2, 0], [0, 2]]


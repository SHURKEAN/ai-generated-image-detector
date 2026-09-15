"""Binary detector metrics and conservative threshold calibration."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _as_arrays(labels: Sequence[float], probabilities: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(probabilities, dtype=np.float64)
    if y_true.shape != y_score.shape or y_true.ndim != 1 or y_true.size == 0:
        raise ValueError("labels and probabilities must be non-empty one-dimensional arrays")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("labels must be 0 (real) or 1 (AI)")
    if not np.isfinite(y_score).all() or ((y_score < 0) | (y_score > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    return y_true, y_score


def binary_metrics(
    labels: Sequence[float],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true, y_score = _as_arrays(labels, probabilities)
    predicted = (y_score >= threshold).astype(np.int64)
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    tn = int(((predicted == 0) & (y_true == 0)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    available_rates = [rate for rate in (recall, specificity) if rate is not None]
    balanced_accuracy = sum(available_rates) / len(available_rates)

    return {
        "threshold": float(threshold),
        "count": int(y_true.size),
        "real_count": int((y_true == 0).sum()),
        "ai_count": int((y_true == 1).sum()),
        "accuracy": float((predicted == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy),
        "precision_ai": precision,
        "recall_ai": recall,
        "specificity_real": specificity,
        "false_positive_rate_real": ratio(fp, fp + tn),
        "f1_ai": f1,
        "confusion_matrix_real_ai": [[tn, fp], [fn, tp]],
    }


def select_threshold(
    labels: Sequence[float],
    probabilities: Sequence[float],
    *,
    maximum_real_false_positive_rate: float = 0.05,
) -> tuple[float, dict[str, Any]]:
    """Choose the best balanced threshold subject to a real-photo FP ceiling."""

    y_true, y_score = _as_arrays(labels, probabilities)
    if not ((y_true == 0).any() and (y_true == 1).any()):
        raise ValueError("Threshold calibration requires both real and AI validation images")
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_score, np.nextafter(y_score, 1.0))))
    eligible: list[tuple[float, dict[str, Any]]] = []
    for threshold in candidates:
        metrics = binary_metrics(y_true, y_score, float(threshold))
        false_positive_rate = metrics["false_positive_rate_real"]
        if false_positive_rate is not None and false_positive_rate <= maximum_real_false_positive_rate:
            eligible.append((float(threshold), metrics))
    if not eligible:
        raise RuntimeError("No threshold satisfied the requested false-positive ceiling")
    return max(
        eligible,
        key=lambda item: (
            item[1]["balanced_accuracy"],
            item[1]["recall_ai"] or 0.0,
            -item[0],
        ),
    )


__all__ = ["binary_metrics", "select_threshold"]


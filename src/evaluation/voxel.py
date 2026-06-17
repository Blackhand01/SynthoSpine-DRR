"""Voxel metrics for early reconstruction experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class VoxelMetrics:
    precision: float
    recall: float
    f1: float
    iou: float

    def as_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "iou": self.iou,
        }


def binarize(volume: NDArray[np.float32], threshold: float = 0.5) -> NDArray[np.bool_]:
    """Convert probability or occupancy arrays to boolean masks."""

    return np.asarray(volume) >= threshold


def voxel_metrics(
    prediction: NDArray[np.float32],
    target: NDArray[np.float32],
    *,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> VoxelMetrics:
    """Compute precision, recall, F1, and IoU for binary volumes."""

    pred = binarize(prediction, threshold)
    truth = binarize(target, threshold)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape}, target {truth.shape}")
    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, ~truth).sum())
    fn = float(np.logical_and(~pred, truth).sum())
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return VoxelMetrics(precision, recall, f1, iou)


def threshold_sweep(
    prediction: NDArray[np.float32],
    target: NDArray[np.float32],
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7),
) -> dict[float, VoxelMetrics]:
    """Evaluate voxel metrics across multiple thresholds."""

    return {threshold: voxel_metrics(prediction, target, threshold=threshold) for threshold in thresholds}


def best_threshold_metrics(
    prediction: NDArray[np.float32],
    target: NDArray[np.float32],
    thresholds: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
) -> dict[str, float]:
    """Return the best F1 over a threshold sweep."""

    scored = [(threshold, voxel_metrics(prediction, target, threshold=threshold)) for threshold in thresholds]
    best_threshold, best_metrics = max(scored, key=lambda item: item[1].f1)
    return {
        "best_threshold": best_threshold,
        "best_f1": best_metrics.f1,
        "best_iou": best_metrics.iou,
    }

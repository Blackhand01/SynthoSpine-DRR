"""Small-volume surface metrics for evaluation smoke runs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SurfaceMetrics:
    surface_score: float
    surface_precision: float
    surface_recall: float
    asd_mm: float
    hd95_mm: float
    hd99_mm: float
    max_surface_distance_mm: float
    nonzero_surface_distance_fraction: float

    def as_dict(self) -> dict[str, float]:
        return {
            "surface_score": self.surface_score,
            "surface_precision": self.surface_precision,
            "surface_recall": self.surface_recall,
            "asd_mm": self.asd_mm,
            "hd95_mm": self.hd95_mm,
            "hd99_mm": self.hd99_mm,
            "max_surface_distance_mm": self.max_surface_distance_mm,
            "nonzero_surface_distance_fraction": self.nonzero_surface_distance_fraction,
        }


def surface_metrics(
    prediction: NDArray[np.float32],
    target: NDArray[np.float32],
    *,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    threshold: float = 0.5,
    tolerance_mm: float = 1.0,
) -> SurfaceMetrics:
    """Compute simple surface metrics using brute-force distances."""

    pred_pts = _surface_points(prediction >= threshold, spacing_mm)
    target_pts = _surface_points(target >= threshold, spacing_mm)
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return SurfaceMetrics(0.0, 0.0, 0.0, float("inf"), float("inf"), float("inf"), float("inf"), 1.0)
    pred_to_target = _min_distances(pred_pts, target_pts)
    target_to_pred = _min_distances(target_pts, pred_pts)
    precision = float((pred_to_target <= tolerance_mm).mean())
    recall = float((target_to_pred <= tolerance_mm).mean())
    score = 2.0 * precision * recall / max(precision + recall, 1e-8)
    all_distances = np.concatenate([pred_to_target, target_to_pred])
    nonzero_fraction = float((all_distances > 0).mean())
    return SurfaceMetrics(
        score,
        precision,
        recall,
        float(all_distances.mean()),
        float(np.percentile(all_distances, 95)),
        float(np.percentile(all_distances, 99)),
        float(all_distances.max()),
        nonzero_fraction,
    )


def _surface_points(mask: NDArray[np.bool_], spacing_mm: tuple[float, float, float]) -> NDArray[np.float32]:
    mask = np.asarray(mask, dtype=bool).squeeze()
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask after squeeze, got {mask.shape}")
    if not mask.any():
        return np.empty((0, 3), dtype=np.float32)
    padded = np.pad(mask, 1, constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    interior = (
        padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return np.argwhere(center & ~interior).astype(np.float32) * np.asarray(spacing_mm, dtype=np.float32)


def _min_distances(source: NDArray[np.float32], target: NDArray[np.float32]) -> NDArray[np.float32]:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(target)
        distances, _ = tree.query(source, k=1, workers=-1)
        return distances.astype(np.float32)
    except Exception:
        return _min_distances_bruteforce(source, target)


def _min_distances_bruteforce(source: NDArray[np.float32], target: NDArray[np.float32]) -> NDArray[np.float32]:
    chunks = []
    for start in range(0, len(source), 512):
        chunk = source[start : start + 512]
        distances = ((chunk[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
        chunks.append(np.sqrt(distances.min(axis=1)))
    return np.concatenate(chunks).astype(np.float32)

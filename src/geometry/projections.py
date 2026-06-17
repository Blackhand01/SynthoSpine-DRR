"""Projection-matrix utilities for geometry-aware reconstruction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.schemas import ProjectionMatrix


def make_intrinsics(focal_px: float, center_x_px: float, center_y_px: float) -> NDArray[np.float32]:
    """Create a pinhole camera intrinsic matrix."""

    return np.array(
        [[focal_px, 0.0, center_x_px], [0.0, focal_px, center_y_px], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def compose_projection(
    intrinsics: NDArray[np.float32],
    rotation: NDArray[np.float32] | None = None,
    translation: NDArray[np.float32] | None = None,
    *,
    source: str = "synthetic",
) -> ProjectionMatrix:
    """Compose P = K [R | t]."""

    rotation = np.eye(3, dtype=np.float32) if rotation is None else rotation.astype(np.float32)
    translation = (
        np.zeros((3, 1), dtype=np.float32)
        if translation is None
        else translation.astype(np.float32).reshape(3, 1)
    )
    extrinsics = np.concatenate([rotation, translation], axis=1)
    return ProjectionMatrix(intrinsics.astype(np.float32) @ extrinsics, source=source)


def adjust_projection_for_crop(
    projection: ProjectionMatrix,
    crop_x_px: float,
    crop_y_px: float,
) -> ProjectionMatrix:
    """Apply crop-adjusted projection P_hat = Q @ P."""

    q = np.array(
        [[1.0, 0.0, -crop_x_px], [0.0, 1.0, -crop_y_px], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return ProjectionMatrix(q @ projection.matrix, projection.coordinate_frame, projection.source)


def project_points(points_xyz: NDArray[np.float32], projection: ProjectionMatrix) -> NDArray[np.float32]:
    """Project Nx3 world points to Nx2 image coordinates."""

    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape [N, 3]")
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    projected = homogeneous @ projection.matrix.T
    return projected[:, :2] / np.clip(projected[:, 2:3], 1e-6, None)


def perturb_projection(
    projection: ProjectionMatrix,
    *,
    focal_delta_px: float = 0.0,
    tx_delta_px: float = 0.0,
    ty_delta_px: float = 0.0,
) -> ProjectionMatrix:
    """Apply simple calibration perturbations for sensitivity analysis."""

    matrix = projection.matrix.copy()
    matrix[0, 0] += focal_delta_px
    matrix[1, 1] += focal_delta_px
    matrix[0, 2] += tx_delta_px
    matrix[1, 2] += ty_delta_px
    return ProjectionMatrix(matrix, projection.coordinate_frame, f"{projection.source}:perturbed")

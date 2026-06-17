"""Lightweight tests for the smoke path."""

from __future__ import annotations

import numpy as np

from ..apps.smoke import run_smoke
from ..evaluation.voxel import voxel_metrics
from ..geometry import adjust_projection_for_crop, compose_projection, make_intrinsics, project_points


def test_projection_crop_shift() -> None:
    intrinsics = make_intrinsics(100.0, 50.0, 60.0)
    projection = compose_projection(intrinsics)
    point = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    before = project_points(point, projection)
    adjusted = adjust_projection_for_crop(projection, 7.0, 11.0)
    after = project_points(point, adjusted)
    assert np.allclose(after - before, [[-7.0, -11.0]])


def test_identity_voxel_metrics() -> None:
    target = np.zeros((1, 8, 8, 8), dtype=np.float32)
    target[:, 2:6, 2:6, 2:6] = 1.0
    metrics = voxel_metrics(target, target)
    assert metrics.f1 > 0.999
    assert metrics.iou > 0.999


def test_smoke_shapes() -> None:
    result = run_smoke(volume_size=16)
    assert result["images_shape"] == [1, 4, 1, 224, 224]
    assert result["projections_shape"] == [1, 4, 3, 4]
    assert result["target_shape"] == [1, 1, 16, 16, 16]
    if result["model_forward"] == "ok":
        assert result["output_shape"] == [1, 1, 16, 16, 16]


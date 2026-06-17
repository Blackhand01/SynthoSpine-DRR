"""Synthetic data generator for geometry and model smoke tests."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..core.config import ImageConfig, VolumeConfig
from ..core.schemas import ReconstructionSample, ViewCategory
from ..geometry import compose_projection, make_intrinsics
from .preprocessing import make_localized_view


def make_ellipsoid_volume(config: VolumeConfig) -> NDArray[np.float32]:
    """Create a deterministic vertebra-like binary occupancy target."""

    z = np.linspace(-1.0, 1.0, config.depth, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, config.height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, config.width, dtype=np.float32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    body = (xx / 0.62) ** 2 + (yy / 0.38) ** 2 + (zz / 0.48) ** 2 < 1.0
    canal = (xx / 0.18) ** 2 + (yy / 0.12) ** 2 < 1.0
    pedicles = ((xx - 0.46) ** 2 + yy**2 < 0.04) | ((xx + 0.46) ** 2 + yy**2 < 0.04)
    volume = (body | (pedicles & (np.abs(zz) < 0.45))) & ~canal
    return volume.astype(np.float32).reshape(1, config.depth, config.height, config.width)


def project_volume_placeholder(volume: NDArray[np.float32], angle_rad: float) -> NDArray[np.float32]:
    """Create a lightweight pseudo-DRR by rotating around the cranio-caudal axis."""

    array = volume[0]
    rotated = _rotate_volume_z(array, angle_rad)
    projection = rotated.sum(axis=0)
    return (projection / max(float(projection.max()), 1.0)).astype(np.float32)


def _rotate_volume_z(volume: NDArray[np.float32], angle_rad: float) -> NDArray[np.float32]:
    angle_deg = math.degrees(angle_rad)
    if abs(angle_deg) < 1e-6:
        return volume
    slices = []
    for slice_2d in volume:
        image = Image.fromarray(slice_2d.astype(np.float32), mode="F")
        rotated = image.rotate(angle_deg, resample=Image.Resampling.BILINEAR)
        slices.append(np.asarray(rotated, dtype=np.float32))
    return np.stack(slices, axis=0)


def make_synthetic_sample(
    *,
    sample_id: str = "synthetic_L3",
    volume_config: VolumeConfig = VolumeConfig(),
    image_config: ImageConfig = ImageConfig(),
) -> ReconstructionSample:
    """Create one four-view sample with valid shapes and projection matrices."""

    target = make_ellipsoid_volume(volume_config)
    categories: tuple[ViewCategory, ...] = ("ap", "lateral", "oblique", "misc")
    angles = (0.0, math.pi / 2, math.pi / 6, math.pi / 4)
    views = []
    intrinsics = make_intrinsics(image_config.width / 2, image_config.width / 2, image_config.height / 2)
    for index, (category, angle) in enumerate(zip(categories, angles, strict=True)):
        projection = compose_projection(intrinsics, source="synthetic_placeholder")
        views.append(
            make_localized_view(
                view_id=f"{sample_id}_{category}_{index:02d}",
                category=category,
                image_2d=_resize_to_square(project_volume_placeholder(target, angle), 256),
                projection=projection,
                bbox_xywh=(16.0, 16.0, 224.0, 224.0),
                output_size=(image_config.height, image_config.width),
            )
        )
    return ReconstructionSample(sample_id, "synthetic_patient", "L3", tuple(views), target, (1.0, 1.0, 1.0))


def _resize_to_square(image: NDArray[np.float32], size: int) -> NDArray[np.float32]:
    y_idx = np.linspace(0, image.shape[0] - 1, size).round().astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, size).round().astype(np.int64)
    return image[y_idx][:, x_idx].astype(np.float32)

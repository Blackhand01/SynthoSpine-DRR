"""Preprocessing helpers for localized X-ray/DRR views."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.schemas import Batch, ProjectionMatrix, ReconstructionSample, ViewCategory, ViewSample
from ..geometry import adjust_projection_for_crop


def normalize_intensity(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize an image to [0, 1] with stable handling of flat inputs."""

    array = np.asarray(image, dtype=np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_value) / (max_value - min_value)).astype(np.float32)


def crop_2d(
    image: NDArray[np.float32],
    bbox_xywh: tuple[float, float, float, float],
) -> NDArray[np.float32]:
    """Crop a 2D image using clamped xywh pixel bounds."""

    x, y, w, h = bbox_xywh
    height, width = image.shape[-2:]
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(width, int(round(x + w)))
    y1 = min(height, int(round(y + h)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid crop {bbox_xywh} for image shape {image.shape}")
    return image[..., y0:y1, x0:x1]


def resize_nearest(image: NDArray[np.float32], height: int, width: int) -> NDArray[np.float32]:
    """Dependency-free nearest-neighbor resize for MVP smoke tests."""

    src_h, src_w = image.shape[-2:]
    y_idx = np.linspace(0, src_h - 1, height).round().astype(np.int64)
    x_idx = np.linspace(0, src_w - 1, width).round().astype(np.int64)
    return image[..., y_idx, :][..., :, x_idx].astype(np.float32)


def make_localized_view(
    *,
    view_id: str,
    category: ViewCategory,
    image_2d: NDArray[np.float32],
    projection: ProjectionMatrix,
    bbox_xywh: tuple[float, float, float, float],
    output_size: tuple[int, int] = (224, 224),
    mask_2d: NDArray[np.float32] | None = None,
) -> ViewSample:
    """Crop, normalize, resize, and update projection metadata."""

    cropped = crop_2d(image_2d, bbox_xywh)
    resized = resize_nearest(normalize_intensity(cropped), *output_size)
    resized = resized.reshape(1, output_size[0], output_size[1])
    mask = None
    if mask_2d is not None:
        mask = resize_nearest(crop_2d(mask_2d, bbox_xywh), *output_size).reshape(resized.shape)
    adjusted = adjust_projection_for_crop(projection, bbox_xywh[0], bbox_xywh[1])
    return ViewSample(view_id, category, resized, adjusted, bbox_xywh, mask)


def stack_batch(samples: list[ReconstructionSample]) -> Batch:
    """Stack reconstruction samples into the shared Batch schema."""

    sample_ids = tuple(sample.sample_id for sample in samples)
    images = np.stack([[view.image for view in sample.views] for sample in samples]).astype(np.float32)
    projections = np.stack(
        [[view.projection.matrix for view in sample.views] for sample in samples]
    ).astype(np.float32)
    targets = np.stack([sample.target_occupancy for sample in samples]).astype(np.float32)
    spacing = np.asarray([sample.spacing_mm for sample in samples], dtype=np.float32)
    return Batch(sample_ids, images, projections, targets, spacing)


"""3D volume helpers for CT-derived reconstruction samples."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .medical_io import resize_volume_nearest


def crop_around_mask(
    volume: NDArray[np.float32],
    mask: NDArray[np.bool_],
    *,
    margin: int = 12,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], tuple[int, int, int, int, int, int]]:
    """Crop a CT/mask pair around nonzero mask voxels."""

    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise ValueError("Cannot crop around an empty mask")
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin + 1, mask.shape)
    z0, y0, x0 = [int(v) for v in mins]
    z1, y1, x1 = [int(v) for v in maxs]
    return volume[z0:z1, y0:y1, x0:x1], mask[z0:z1, y0:y1, x0:x1], (z0, z1, y0, y1, x0, x1)


def resize_pair(
    volume: NDArray[np.float32],
    mask: NDArray[np.bool_],
    shape: tuple[int, int, int],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Resize CT intensity and binary mask to a shared grid."""

    resized_volume = resize_volume_nearest(volume.astype(np.float32), shape)
    resized_mask = resize_volume_nearest(mask.astype(np.float32), shape)
    return normalize_ct(resized_volume), (resized_mask >= 0.5).astype(np.float32)


def normalize_ct(volume: NDArray[np.float32]) -> NDArray[np.float32]:
    """Robustly normalize CT intensities to [0, 1] for proxy DRR generation."""

    array = np.asarray(volume, dtype=np.float32)
    low, high = np.percentile(array, (1.0, 99.0))
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    clipped = np.clip(array, low, high)
    return ((clipped - low) / (high - low)).astype(np.float32)

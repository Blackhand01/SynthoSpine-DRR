"""Qualitative visualization helpers for MVP evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def save_slice_overlay(
    prediction: NDArray[np.float32],
    target: NDArray[np.float32],
    output_path: Path,
    *,
    threshold: float = 0.5,
) -> None:
    """Save a central axial overlay: target green, prediction red, overlap yellow."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(prediction).squeeze() >= threshold
    truth = np.asarray(target).squeeze() >= threshold
    if pred.shape != truth.shape or pred.ndim != 3:
        raise ValueError("Prediction and target must be matching 3D volumes")
    z_index = _best_axis_index(pred | truth, axis=0)
    image = np.zeros((*pred[z_index].shape, 3), dtype=np.uint8)
    image[..., 0] = pred[z_index].astype(np.uint8) * 255
    image[..., 1] = truth[z_index].astype(np.uint8) * 255
    _write_png(image, output_path)


def save_volume_slices(volume: NDArray[np.float32], output_path: Path, *, threshold: float = 0.5) -> None:
    """Save axial/sagittal/coronal central slices as a single PNG strip."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask = (np.asarray(volume).squeeze() >= threshold).astype(np.uint8) * 255
    if mask.ndim != 3:
        raise ValueError("Volume must be 3D after squeeze")
    filled = mask > 0
    z_index = _best_axis_index(filled, axis=0)
    x_index = _best_axis_index(filled, axis=2)
    y_index = _best_axis_index(filled, axis=1)
    slices = [mask[z_index], mask[:, :, x_index], mask[:, y_index, :]]
    strip = _pad_and_concat(slices)
    _write_png(np.repeat(strip[..., None], 3, axis=2), output_path)


def save_voxel_npz(prediction: NDArray[np.float32], target: NDArray[np.float32], output_path: Path) -> None:
    """Save prediction probabilities and target occupancy for later inspection."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, prediction=prediction.astype(np.float32), target=target.astype(np.float32))


def save_surface_ply(volume: NDArray[np.float32], output_path: Path, *, threshold: float = 0.5) -> None:
    """Save exposed voxel centers as an ASCII PLY point cloud."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = _surface_points(np.asarray(volume).squeeze() >= threshold)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("end_header\n")
        for z, y, x in points:
            handle.write(f"{x:.3f} {y:.3f} {z:.3f}\n")


def save_input_views(images: NDArray[np.float32], output_dir: Path) -> None:
    """Save localized 2D DRR input views as individual PNGs and one strip."""

    output_dir.mkdir(parents=True, exist_ok=True)
    names = ("ap", "lateral", "oblique", "misc")
    views = np.asarray(images).squeeze(axis=0)
    pngs = []
    for index, view in enumerate(views):
        image = _to_uint8(view.squeeze())
        path = output_dir / f"view_{index:02d}_{names[index] if index < len(names) else 'view'}.png"
        _write_png(np.repeat(image[..., None], 3, axis=2), path)
        pngs.append(image)
    strip = _pad_and_concat(pngs)
    _write_png(np.repeat(strip[..., None], 3, axis=2), output_dir / "views_strip.png")


def _pad_and_concat(images: list[NDArray[np.uint8]]) -> NDArray[np.uint8]:
    height = max(image.shape[0] for image in images)
    padded = [np.pad(image, ((0, height - image.shape[0]), (0, 0)), constant_values=0) for image in images]
    return np.concatenate(padded, axis=1)


def _write_png(image: NDArray[np.uint8], output_path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write PNG visualizations") from exc
    Image.fromarray(image).save(output_path)


def _to_uint8(image: NDArray[np.float32]) -> NDArray[np.uint8]:
    array = np.asarray(image, dtype=np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint8)
    return ((array - min_value) / (max_value - min_value) * 255).astype(np.uint8)


def _best_axis_index(mask: NDArray[np.bool_], axis: int) -> int:
    counts = mask.sum(axis=tuple(index for index in range(mask.ndim) if index != axis))
    if counts.max() == 0:
        return mask.shape[axis] // 2
    return int(np.argmax(counts))


def _surface_points(mask: NDArray[np.bool_]) -> NDArray[np.float32]:
    if mask.ndim != 3:
        raise ValueError("Volume must be 3D after squeeze")
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
    return np.argwhere(center & ~interior).astype(np.float32)

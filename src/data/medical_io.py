"""Medical volume IO with optional SimpleITK support."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.schemas import VolumeRecord


def load_volume_pair(ct_path: str | Path, mask_path: str | Path, patient_id: str | None = None) -> VolumeRecord:
    """Load a CT/mask pair from NPZ or SimpleITK-readable files."""

    ct = Path(ct_path)
    mask = Path(mask_path)
    if ct.suffix == ".npz" and mask.suffix == ".npz":
        volume_npz = np.load(ct)
        mask_npz = np.load(mask)
        volume = volume_npz["volume"].astype(np.float32)
        mask_array = mask_npz["mask"].astype(np.float32)
        spacing = tuple(float(x) for x in volume_npz.get("spacing_mm", (1.0, 1.0, 1.0)))
    else:
        volume, spacing = _load_sitk_array(ct)
        mask_array, _ = _load_sitk_array(mask)
    if volume.shape != mask_array.shape:
        raise ValueError(f"CT/mask shape mismatch: {volume.shape} vs {mask_array.shape}")
    return VolumeRecord(patient_id or ct.stem, volume, mask_array, spacing)


def _load_sitk_array(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("Install SimpleITK or use .npz fixtures for medical IO") from exc
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = tuple(float(x) for x in image.GetSpacing())
    return array, (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])


def resize_volume_nearest(volume: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Resize a 3D array with dependency-free nearest-neighbor sampling."""

    z_idx = np.linspace(0, volume.shape[0] - 1, shape[0]).round().astype(np.int64)
    y_idx = np.linspace(0, volume.shape[1] - 1, shape[1]).round().astype(np.int64)
    x_idx = np.linspace(0, volume.shape[2] - 1, shape[2]).round().astype(np.int64)
    return volume[z_idx][:, y_idx][:, :, x_idx].astype(np.float32)


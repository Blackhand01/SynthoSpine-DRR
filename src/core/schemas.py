"""Typed data contracts shared by the prototype modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ViewCategory = Literal["ap", "lateral", "oblique", "misc"]


@dataclass(frozen=True)
class ProjectionMatrix:
    """A calibrated or synthetic 3x4 projection matrix."""

    matrix: NDArray[np.float32]
    coordinate_frame: str = "world"
    source: str = "synthetic"

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float32)
        if matrix.shape != (3, 4):
            raise ValueError(f"Projection matrix must be 3x4, got {matrix.shape}")
        object.__setattr__(self, "matrix", matrix)


@dataclass(frozen=True)
class ViewSample:
    """One localized 2D view of the anatomy."""

    view_id: str
    category: ViewCategory
    image: NDArray[np.float32]
    projection: ProjectionMatrix
    bbox_xywh: tuple[float, float, float, float]
    mask_2d: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        image = np.asarray(self.image, dtype=np.float32)
        if image.ndim != 3 or image.shape[0] != 1:
            raise ValueError("View image must have shape [1, H, W]")
        if self.mask_2d is not None and self.mask_2d.shape[-2:] != image.shape[-2:]:
            raise ValueError("2D mask spatial shape must match image")
        object.__setattr__(self, "image", image)


@dataclass(frozen=True)
class ReconstructionSample:
    """Full multi-view sample used by the model and metrics."""

    sample_id: str
    patient_id: str
    anatomy_id: str
    views: tuple[ViewSample, ...]
    target_occupancy: NDArray[np.float32]
    spacing_mm: tuple[float, float, float]

    def __post_init__(self) -> None:
        target = np.asarray(self.target_occupancy, dtype=np.float32)
        if target.ndim != 4 or target.shape[0] != 1:
            raise ValueError("Target occupancy must have shape [1, D, H, W]")
        if not self.views:
            raise ValueError("A reconstruction sample needs at least one view")
        object.__setattr__(self, "target_occupancy", target)


@dataclass(frozen=True)
class Batch:
    """Batched arrays before conversion to torch."""

    sample_ids: tuple[str, ...]
    images: NDArray[np.float32]
    projections: NDArray[np.float32]
    target_occupancy: NDArray[np.float32]
    spacing_mm: NDArray[np.float32]


@dataclass(frozen=True)
class VolumeRecord:
    """Loaded medical volume and segmentation pair."""

    patient_id: str
    volume: NDArray[np.float32]
    mask: NDArray[np.float32]
    spacing_mm: tuple[float, float, float]


@dataclass(frozen=True)
class ManifestRecord:
    """Serializable CTSpine/DRR sample metadata."""

    sample_id: str
    patient_id: str
    anatomy_id: str
    ct_path: str
    mask_path: str
    sample_path: str
    spacing_mm: tuple[float, float, float]
    views: tuple[str, ...]

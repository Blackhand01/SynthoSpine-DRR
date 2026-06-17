"""PyTorch datasets for generated reconstruction samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.manifest import read_manifest
from ..data.preprocessing import stack_batch
from ..data.synthetic import make_synthetic_sample
from ..core.config import VolumeConfig


class NpzReconstructionDataset:
    """Dataset backed by generated NPZ samples."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.records = [record for record in read_manifest(manifest_path) if Path(record["sample_path"]).exists()]
        if not self.records:
            raise ValueError(f"No generated samples found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        data = np.load(record["sample_path"])
        return {
            "sample_id": str(data["sample_id"]),
            "images": data["images"].astype(np.float32),
            "projections": data["projections"].astype(np.float32),
            "target": data["target"].astype(np.float32),
            "spacing_mm": data["spacing_mm"].astype(np.float32),
        }


def make_manifest_training_item(manifest_path: str | Path, index: int = 0) -> dict:
    """Return one generated manifest item for overfit/evaluation."""

    dataset = NpzReconstructionDataset(manifest_path)
    return dataset[index]


def make_synthetic_training_item(volume_size: int = 16) -> dict:
    """Return one synthetic item matching NpzReconstructionDataset output."""

    sample = make_synthetic_sample(volume_config=VolumeConfig(volume_size, volume_size, volume_size))
    batch = stack_batch([sample])
    return {
        "sample_id": sample.sample_id,
        "images": batch.images[0],
        "projections": batch.projections[0],
        "target": batch.target_occupancy[0],
        "spacing_mm": batch.spacing_mm[0],
    }


def collate_items(items: list[dict]) -> dict:
    """Collate dict items into numpy batch arrays."""

    return {
        "sample_id": tuple(item["sample_id"] for item in items),
        "images": np.stack([item["images"] for item in items]).astype(np.float32),
        "projections": np.stack([item["projections"] for item in items]).astype(np.float32),
        "target": np.stack([item["target"] for item in items]).astype(np.float32),
        "spacing_mm": np.stack([item["spacing_mm"] for item in items]).astype(np.float32),
    }

"""Generate small CT-derived DRR proxy samples."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np

from ..core.config import ImageConfig, VolumeConfig
from ..core.config_io import load_config
from ..data.manifest import DEFAULTS, read_manifest
from ..data.medical_io import load_volume_pair, resize_volume_nearest
from ..data.volume_ops import crop_around_mask, resize_pair
from ..data.preprocessing import stack_batch
from ..data.synthetic import project_volume_placeholder
from ..geometry import compose_projection, make_intrinsics
from ..core.schemas import ReconstructionSample
from ..data.preprocessing import make_localized_view

VERTEBRA_LABELS = {
    "C1": 1,
    "C2": 2,
    "C3": 3,
    "C4": 4,
    "C5": 5,
    "C6": 6,
    "C7": 7,
    "T1": 8,
    "T2": 9,
    "T3": 10,
    "T4": 11,
    "T5": 12,
    "T6": 13,
    "T7": 14,
    "T8": 15,
    "T9": 16,
    "T10": 17,
    "T11": 18,
    "T12": 19,
    "L1": 20,
    "L2": 21,
    "L3": 22,
    "L4": 23,
    "L5": 24,
}


def generate_record_sample(record: dict, crop_size: int = 128, volume_size: int = 64) -> dict:
    """Generate and save one manifest-backed DRR proxy sample."""

    start = time.perf_counter()
    volume = load_volume_pair(record["ct_path"], record["mask_path"], record["patient_id"])
    label = _label_for_anatomy(record["anatomy_id"])
    raw_mask = volume.mask == label
    if not raw_mask.any():
        raise ValueError(f"Label {label} for {record['anatomy_id']} is empty in {record['patient_id']}")
    crop_volume, crop_mask, crop_bbox = crop_around_mask(volume.volume, raw_mask)
    input_volume, target = resize_pair(crop_volume, crop_mask, (volume_size, volume_size, volume_size))
    target = target.astype(np.float32).reshape(1, volume_size, volume_size, volume_size)
    sample = _make_sample_from_target(record, target, crop_size, input_volume.reshape(target.shape))
    batch = stack_batch([sample])
    output = Path(record["sample_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        images=batch.images[0],
        projections=batch.projections[0],
        target=batch.target_occupancy[0],
        spacing_mm=np.asarray(volume.spacing_mm, dtype=np.float32),
        sample_id=record["sample_id"],
        patient_id=record["patient_id"],
        vertebra_label=np.asarray(label, dtype=np.int16),
        crop_bbox_zyx=np.asarray(crop_bbox, dtype=np.int32),
    )
    elapsed = time.perf_counter() - start
    del volume, raw_mask, crop_volume, crop_mask, input_volume, target, sample, batch
    gc.collect()
    return {"sample_id": record["sample_id"], "sample_path": str(output), "drr_seconds": elapsed}


def generate_samples(manifest_path: str | Path, crop_size: int = 128, volume_size: int = 64, limit: int | None = None) -> list[dict]:
    records = read_manifest(manifest_path)
    outputs = []
    for record in records[:limit]:
        outputs.append(generate_record_sample(record, crop_size, volume_size))
    return outputs


def _label_for_anatomy(anatomy_id: str) -> int:
    key = anatomy_id.upper()
    if key not in VERTEBRA_LABELS:
        raise ValueError(f"Unsupported anatomy_id {anatomy_id!r}; expected one of {sorted(VERTEBRA_LABELS)}")
    return VERTEBRA_LABELS[key]


def _make_sample_from_target(
    record: dict,
    target: np.ndarray,
    crop_size: int,
    input_volume: np.ndarray | None = None,
) -> ReconstructionSample:
    categories = tuple(record.get("views", ("ap", "lateral", "oblique", "misc")))
    angles = {"ap": 0.0, "lateral": np.pi / 2, "oblique": np.pi / 6, "misc": np.pi / 4}
    intrinsics = make_intrinsics(crop_size / 2, crop_size / 2, crop_size / 2)
    views = []
    source_volume = input_volume if input_volume is not None else target
    for idx, category in enumerate(categories):
        image = _resize_square(project_volume_placeholder(source_volume, angles.get(category, 0.0)), crop_size + 32)
        projection = compose_projection(intrinsics, source="ct_drr_proxy")
        views.append(
            make_localized_view(
                view_id=f"{record['sample_id']}_{category}_{idx:02d}",
                category=category,
                image_2d=image,
                projection=projection,
                bbox_xywh=(16.0, 16.0, float(crop_size), float(crop_size)),
                output_size=(crop_size, crop_size),
            )
        )
    return ReconstructionSample(record["sample_id"], record["patient_id"], record["anatomy_id"], tuple(views), target, tuple(record["spacing_mm"]))


def _resize_square(image: np.ndarray, size: int) -> np.ndarray:
    y_idx = np.linspace(0, image.shape[0] - 1, size).round().astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, size).round().astype(np.int64)
    return image[y_idx][:, x_idx].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CT-derived DRR proxy samples.")
    parser.add_argument("--config", default="configs/micro_ctspine.yaml")
    parser.add_argument("--manifest", default="data/processed/micro_ctspine/manifest.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config, DEFAULTS | {"crop_size": 128, "volume_size": 64})
    outputs = generate_samples(args.manifest, int(config["crop_size"]), int(config["volume_size"]), args.limit)
    print(f"Generated {len(outputs)} DRR proxy samples")


if __name__ == "__main__":
    main()

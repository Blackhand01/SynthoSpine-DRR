"""Command-line smoke checks for the Spine2Space MVP."""

from __future__ import annotations

import argparse
import json

import numpy as np

from ..core.config import ModelConfig, VolumeConfig
from ..data import make_synthetic_sample, stack_batch
from ..evaluation.voxel import voxel_metrics
from ..geometry import adjust_projection_for_crop, project_points


def run_smoke(volume_size: int = 24) -> dict:
    """Run data, geometry, metric, and model smoke checks."""

    sample = make_synthetic_sample(volume_config=VolumeConfig(volume_size, volume_size, volume_size))
    batch = stack_batch([sample])
    result = {
        "sample_id": sample.sample_id,
        "images_shape": list(batch.images.shape),
        "projections_shape": list(batch.projections.shape),
        "target_shape": list(batch.target_occupancy.shape),
        "identity_metrics": voxel_metrics(sample.target_occupancy, sample.target_occupancy).as_dict(),
        "crop_shift_px": _check_crop_shift(),
    }
    result.update(_run_model_forward(batch, volume_size))
    return result


def _check_crop_shift() -> list[float]:
    sample = make_synthetic_sample()
    view = sample.views[0]
    point = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    before = project_points(point, view.projection)
    shifted = adjust_projection_for_crop(view.projection, 10.0, 20.0)
    after = project_points(point, shifted)
    return [float(after[0, 0] - before[0, 0]), float(after[0, 1] - before[0, 1])]


def _run_model_forward(batch, volume_size: int) -> dict:
    try:
        import torch

        from ..models import SparseXrayTo3D, count_parameters
    except ImportError as exc:
        return {"model_forward": "skipped", "reason": str(exc)}

    config = ModelConfig(volume=VolumeConfig(volume_size, volume_size, volume_size))
    model = SparseXrayTo3D(config)
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(batch.images), torch.from_numpy(batch.projections))
    return {"model_forward": "ok", "output_shape": list(output.shape), "parameter_count": count_parameters(model)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Spine2Space smoke checks.")
    parser.add_argument("--volume-size", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.volume_size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


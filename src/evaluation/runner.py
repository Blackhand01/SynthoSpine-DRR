"""End-to-end synthetic evaluation runner."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..core.config import ModelConfig, VolumeConfig
from ..data import make_synthetic_sample, stack_batch
from ..training.dataset import collate_items, make_manifest_training_item
from ..evaluation.reporting import write_json, write_markdown_report, write_threshold_csv
from ..evaluation.surface import surface_metrics
from ..evaluation.visualization import save_input_views, save_slice_overlay, save_surface_ply, save_volume_slices, save_voxel_npz
from ..evaluation.voxel import threshold_sweep, voxel_metrics


def run_evaluation(
    output_dir: Path,
    volume_size: int = 24,
    checkpoint: str | Path | None = None,
    manifest_path: str | Path | None = None,
    sample_index: int = 0,
) -> dict:
    """Run synthetic model evaluation and persist artifacts."""

    batch, sample_id, spacing = _make_eval_batch(volume_size, manifest_path, sample_index)
    volume_size = int(batch.target_occupancy.shape[-1])
    prediction, model_status = _predict(batch, volume_size, checkpoint)
    target = batch.target_occupancy
    metrics = voxel_metrics(prediction, target)
    surface = surface_metrics(prediction, target, spacing_mm=spacing)
    sweep = [
        {"threshold": threshold, **values.as_dict()}
        for threshold, values in threshold_sweep(prediction, target).items()
    ]
    summary = {
        "sample_id": sample_id,
        "volume_size": volume_size,
        "model_forward": model_status,
        "prediction_shape": list(prediction.shape),
        "target_shape": list(target.shape),
        "metrics": metrics.as_dict(),
        "surface_metrics": surface.as_dict(),
    }
    _write_artifacts(output_dir, summary, sweep, prediction, target, batch.images)
    return summary


def _make_eval_batch(volume_size: int, manifest_path: str | Path | None, sample_index: int):
    if manifest_path is not None:
        item = make_manifest_training_item(manifest_path, sample_index)
        batch_dict = collate_items([item])
        from ..core.schemas import Batch

        batch = Batch(batch_dict["sample_id"], batch_dict["images"], batch_dict["projections"], batch_dict["target"], batch_dict["spacing_mm"])
        return batch, item["sample_id"], tuple(float(x) for x in item["spacing_mm"])
    sample = make_synthetic_sample(volume_config=VolumeConfig(volume_size, volume_size, volume_size))
    return stack_batch([sample]), sample.sample_id, sample.spacing_mm


def _predict(batch, volume_size: int, checkpoint: str | Path | None = None) -> tuple[np.ndarray, str]:
    try:
        import torch

        from ..models import SparseXrayTo3D
    except ImportError:
        return batch.target_occupancy.copy(), "skipped_torch_unavailable"

    config = ModelConfig(volume=VolumeConfig(volume_size, volume_size, volume_size))
    model = SparseXrayTo3D(config)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state"])
    model.eval()
    with torch.no_grad():
        output = model(torch.from_numpy(batch.images), torch.from_numpy(batch.projections))
    return output.numpy().astype(np.float32), "ok_checkpoint" if checkpoint else "ok"


def _write_artifacts(output_dir: Path, summary: dict, sweep: list[dict], prediction: np.ndarray, target: np.ndarray, images: np.ndarray) -> None:
    write_json(summary, output_dir / "metrics.json")
    write_threshold_csv(sweep, output_dir / "threshold_sweep.csv")
    write_markdown_report(summary, output_dir / "report.md")
    save_slice_overlay(prediction, target, output_dir / "qualitative" / "overlay_axial.png")
    save_volume_slices(target, output_dir / "qualitative" / "target_slices.png")
    save_volume_slices(prediction, output_dir / "qualitative" / "prediction_slices.png")
    save_voxel_npz(prediction, target, output_dir / "volumes" / "prediction_target.npz")
    save_surface_ply(prediction, output_dir / "meshes" / "prediction_surface.ply")
    save_surface_ply(target, output_dir / "meshes" / "target_surface.ply")
    save_input_views(images, output_dir / "input_views")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Spine2Space synthetic evaluation.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/day_10_12_eval"))
    parser.add_argument("--volume-size", type=int, default=24)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    summary = run_evaluation(args.output_dir, args.volume_size, args.checkpoint, args.manifest, args.sample_index)
    print(f"Wrote evaluation artifacts to {args.output_dir}")
    print(f"F1={summary['metrics']['f1']:.4f} IoU={summary['metrics']['iou']:.4f}")


if __name__ == "__main__":
    main()

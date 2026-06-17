"""Micro-subset training command."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from ..core.config import ModelConfig, VolumeConfig
from ..core.config_io import load_config
from ..evaluation.reporting import write_json
from ..evaluation.voxel import best_threshold_metrics, voxel_metrics
from ..models import SparseXrayTo3D
from .dataset import NpzReconstructionDataset, collate_items
from .losses import balanced_bce_dice_loss
from .plots import write_metric_plots
from .telemetry import cleanup, collect_runtime, now


DEFAULTS = {"manifest_path": "data/processed/micro_ctspine/manifest.jsonl", "output_dir": "runs/subset_train", "volume_size": 32, "epochs": 1, "learning_rate": 0.001}


def run_subset_training(config: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NpzReconstructionDataset(config["manifest_path"])
    first_item = dataset[0]
    size = int(first_item["target"].shape[-1])
    model = SparseXrayTo3D(ModelConfig(volume=VolumeConfig(size, size, size))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        for index in range(len(dataset)):
            start = now()
            batch = collate_items([dataset[index]])
            images = torch.from_numpy(batch["images"]).to(device)
            projections = torch.from_numpy(batch["projections"]).to(device)
            target = torch.from_numpy(batch["target"]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(images, projections)
            loss = balanced_bce_dice_loss(prediction, target)
            loss.backward()
            optimizer.step()
            pred_np = prediction.detach().cpu().numpy()
            target_np = target.cpu().numpy()
            metrics = voxel_metrics(pred_np, target_np).as_dict()
            metrics.update(best_threshold_metrics(pred_np, target_np))
            history.append({"epoch": epoch, "step": len(history) + 1, "loss": float(loss.detach().cpu()), **metrics, **collect_runtime(start)})
            cleanup()
    return _save_run(config, model, history)


def _save_run(config: dict, model: torch.nn.Module, history: list[dict]) -> dict:
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config}, output / "best.pt")
    write_json({"history": history, "final": history[-1]}, output / "metrics.json")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    write_metric_plots(history, output / "reports" / "figures")
    return history[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on generated micro CTSpine samples.")
    parser.add_argument("--config", default="configs/subset_train.yaml")
    args = parser.parse_args()
    final = run_subset_training(load_config(args.config, DEFAULTS))
    print(f"Final loss={final['loss']:.4f} F1={final['f1']:.4f} BestF1={final.get('best_f1', 0.0):.4f}")


if __name__ == "__main__":
    main()

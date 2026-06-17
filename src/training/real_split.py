"""Patient-level train/validation loop for CT-derived samples."""

from __future__ import annotations

import argparse
import csv
import copy
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


DEFAULTS = {
    "train_manifest": "data/processed/real_split/manifest_train.jsonl",
    "val_manifest": "data/processed/real_split/manifest_val.jsonl",
    "output_dir": "runs/real_split_train",
    "epochs": 20,
    "learning_rate": 0.001,
}


def run_real_split_training(config: dict) -> dict:
    """Train on one patient split and validate on held-out patients."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = NpzReconstructionDataset(config["train_manifest"])
    val_data = NpzReconstructionDataset(config["val_manifest"])
    size = int(train_data[0]["target"].shape[-1])
    model = SparseXrayTo3D(ModelConfig(volume=VolumeConfig(size, size, size))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    history, best = [], {"val_f1": -1.0, "state": None}
    for epoch in range(1, int(config["epochs"]) + 1):
        train_row = _run_epoch(model, train_data, device, optimizer)
        val_row = _run_epoch(model, val_data, device, None)
        row = _merge_rows(epoch, train_row, val_row)
        history.append(row)
        if row["val_f1"] > best["val_f1"]:
            best = {"val_f1": row["val_f1"], "state": copy.deepcopy(model.state_dict()), "row": row}
        cleanup()
    return _save_run(config, model, best, history)


def _run_epoch(model, dataset, device, optimizer) -> dict:
    training = optimizer is not None
    model.train(training)
    rows = []
    for index in range(len(dataset)):
        start = now()
        batch = collate_items([dataset[index]])
        images = torch.from_numpy(batch["images"]).to(device)
        projections = torch.from_numpy(batch["projections"]).to(device)
        target = torch.from_numpy(batch["target"]).to(device)
        with torch.set_grad_enabled(training):
            prediction = model(images, projections)
            loss = balanced_bce_dice_loss(prediction, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        rows.append(_metrics_row(prediction, target, loss, start))
        cleanup()
    return _mean_row(rows)


def _metrics_row(prediction, target, loss, start: float) -> dict:
    pred_np = prediction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    metrics = voxel_metrics(pred_np, target_np).as_dict()
    metrics.update(best_threshold_metrics(pred_np, target_np))
    return {"loss": float(loss.detach().cpu()), **metrics, **collect_runtime(start)}


def _mean_row(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def _merge_rows(epoch: int, train: dict, val: dict) -> dict:
    row = {"epoch": epoch, "step": epoch}
    row.update({f"train_{key}": value for key, value in train.items()})
    row.update({f"val_{key}": value for key, value in val.items()})
    row.update({"loss": train["loss"], "f1": val["f1"], "iou": val["iou"], "seconds": train["seconds"] + val["seconds"], "ram_mb": max(train["ram_mb"], val["ram_mb"]), "vram_mb": max(train["vram_mb"], val["vram_mb"])})
    return row


def _save_run(config: dict, model, best: dict, history: list[dict]) -> dict:
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    state = best["state"] or model.state_dict()
    torch.save({"model_state": state, "config": config}, output / "best.pt")
    write_json({"history": history, "final": history[-1], "best": best.get("row", history[-1])}, output / "metrics.json")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    write_metric_plots(history, output / "reports" / "figures")
    return best.get("row", history[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train with patient-level validation split.")
    parser.add_argument("--config", default="configs/kaggle_real_split.yaml")
    args = parser.parse_args()
    final = run_real_split_training(load_config(args.config, DEFAULTS))
    print(f"Best ValF1={final['val_f1']:.4f} ValIoU={final['val_iou']:.4f} TrainF1={final['train_f1']:.4f}")


if __name__ == "__main__":
    main()

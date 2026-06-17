"""One-sample overfit training command."""

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
from .dataset import make_manifest_training_item, make_synthetic_training_item
from .losses import balanced_bce_dice_loss, bce_dice_loss
from .plots import write_metric_plots
from .telemetry import cleanup, collect_runtime, now


DEFAULTS = {"output_dir": "runs/overfit_synthetic", "volume_size": 16, "steps": 80, "learning_rate": 0.003, "target_f1": 0.90, "loss": "bce_dice"}


def run_overfit(config: dict) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    item = _load_item(config)
    volume_size = int(item["target"].shape[-1])
    images = torch.from_numpy(item["images"][None]).to(device)
    projections = torch.from_numpy(item["projections"][None]).to(device)
    target = torch.from_numpy(item["target"][None]).to(device)
    model = SparseXrayTo3D(ModelConfig(volume=VolumeConfig(volume_size, volume_size, volume_size))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    history = []
    for step in range(1, int(config["steps"]) + 1):
        start = now()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(images, projections)
        loss = _loss(prediction, target, str(config.get("loss", "bce_dice")))
        loss.backward()
        optimizer.step()
        pred_np = prediction.detach().cpu().numpy()
        target_np = target.cpu().numpy()
        metrics = voxel_metrics(pred_np, target_np).as_dict()
        metrics.update(best_threshold_metrics(pred_np, target_np))
        row = {"step": step, "loss": float(loss.detach().cpu()), **metrics, **collect_runtime(start)}
        history.append(row)
        cleanup()
        if row["f1"] >= float(config["target_f1"]):
            break
    return _save_run(config, model, history)


def _load_item(config: dict) -> dict:
    if config.get("manifest_path"):
        return make_manifest_training_item(config["manifest_path"], int(config.get("sample_index", 0)))
    return make_synthetic_training_item(int(config["volume_size"]))


def _loss(prediction: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "balanced":
        return balanced_bce_dice_loss(prediction, target)
    return bce_dice_loss(prediction, target)


def _save_run(config: dict, model: torch.nn.Module, history: list[dict]) -> dict:
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config}, output / "best.pt")
    write_json({"history": history, "final": history[-1]}, output / "metrics.json")
    _write_csv(history, output / "history.csv")
    write_metric_plots(history, output / "reports" / "figures")
    return history[-1]


def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit one synthetic reconstruction sample.")
    parser.add_argument("--config", default="configs/overfit.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config, DEFAULTS)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    final = run_overfit(config)
    print(f"Final loss={final['loss']:.4f} F1={final['f1']:.4f} BestF1={final.get('best_f1', 0.0):.4f}")


if __name__ == "__main__":
    main()

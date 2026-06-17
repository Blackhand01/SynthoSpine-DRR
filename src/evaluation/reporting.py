"""Experiment report writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_threshold_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = summary["metrics"]
    surface = summary["surface_metrics"]
    lines = [
        "# Spine2Space Evaluation Report",
        "",
        f"- Sample: `{summary['sample_id']}`",
        f"- Volume size: `{summary['volume_size']}`",
        f"- Model forward: `{summary['model_forward']}`",
        "",
        "## Voxel Metrics",
        "",
        f"- Precision: `{metrics['precision']:.4f}`",
        f"- Recall: `{metrics['recall']:.4f}`",
        f"- F1: `{metrics['f1']:.4f}`",
        f"- IoU: `{metrics['iou']:.4f}`",
        "",
        "## Surface Metrics",
        "",
        f"- Surface score: `{surface['surface_score']:.4f}`",
        f"- ASD mm: `{surface['asd_mm']:.4f}`",
        f"- HD95 mm: `{surface['hd95_mm']:.4f}`",
        f"- HD99 mm: `{surface.get('hd99_mm', 0.0):.4f}`",
        f"- Max surface distance mm: `{surface.get('max_surface_distance_mm', 0.0):.4f}`",
        f"- Nonzero surface distance fraction: `{surface.get('nonzero_surface_distance_fraction', 0.0):.4f}`",
        "",
        "## Notes",
        "",
        "- Synthetic placeholder data validates contracts, geometry, metrics, and forward pass.",
        "- Full clinical claims require CT/DRR integration, real calibration data, and patient-level splits.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

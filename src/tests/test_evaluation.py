"""Evaluation tests runnable with pytest."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..evaluation.runner import run_evaluation
from ..evaluation.surface import surface_metrics


def test_surface_identity() -> None:
    volume = np.zeros((1, 12, 12, 12), dtype=np.float32)
    volume[:, 3:9, 3:9, 3:9] = 1.0
    metrics = surface_metrics(volume, volume, tolerance_mm=0.1)
    assert metrics.surface_score > 0.999
    assert metrics.asd_mm == 0.0
    assert metrics.hd95_mm == 0.0


def test_evaluation_artifacts(tmp_path: Path) -> None:
    summary = run_evaluation(tmp_path, volume_size=12)
    assert summary["prediction_shape"] == [1, 1, 12, 12, 12]
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "threshold_sweep.csv").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "qualitative" / "overlay_axial.png").exists()


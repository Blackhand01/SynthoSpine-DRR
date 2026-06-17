"""PNG plot helpers for training dashboards."""

from __future__ import annotations

from pathlib import Path


def write_metric_plots(history: list[dict], output_dir: str | Path) -> None:
    """Write loss/F1 and runtime plots if matplotlib is available."""

    if not history:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    steps = [row.get("step", row.get("epoch", idx)) for idx, row in enumerate(history)]
    _plot(plt, output / "clinical_metrics.png", steps, history, ["loss", "f1", "iou"])
    _plot(plt, output / "engineering_metrics.png", steps, history, ["seconds", "ram_mb", "vram_mb"])


def _plot(plt, path: Path, steps: list, history: list[dict], keys: list[str]) -> None:
    plt.figure(figsize=(8, 4))
    for key in keys:
        values = [row.get(key) for row in history if key in row]
        x_values = steps[: len(values)]
        if values:
            plt.plot(x_values, values, label=key)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


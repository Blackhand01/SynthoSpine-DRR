"""3D recruiter visualizations for predicted and target volumes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def export_rotation_gif(
    npz_path: Path,
    output_path: Path,
    *,
    threshold: float = 0.5,
    frames: int = 48,
) -> None:
    """Write a rotating GIF showing overlap, misses, and false positives."""

    try:
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib and imageio are required for 3D GIF export") from exc

    data = _load_masks(npz_path, threshold)
    points = _classified_surface_points(data["prediction"], data["target"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = []
    for index in range(frames):
        fig = plt.figure(figsize=(8, 6), dpi=130)
        ax = fig.add_subplot(111, projection="3d")
        _draw_scene(ax, points, azim=360.0 * index / frames)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        rendered.append(rgba[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(output_path, rendered, duration=0.08)


def export_interactive_html(npz_path: Path, output_path: Path, *, threshold: float = 0.5) -> None:
    """Write an interactive Plotly viewer for reconstruction inspection."""

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("plotly is required for HTML 3D export") from exc

    data = _load_masks(npz_path, threshold)
    points = _classified_surface_points(data["prediction"], data["target"])
    traces = []
    for name, color, size in (
        ("Overlap: prediction and ground truth", "#f2c94c", 4),
        ("False negatives: missed target", "#2d9cdb", 5),
        ("False positives: extra prediction", "#eb5757", 5),
    ):
        xyz = points[name]
        traces.append(_plotly_trace(go, xyz, name, color, size))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Spine2Space 3D Reconstruction - Prediction vs Ground Truth",
        scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(x=0.02, y=0.98),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")


def _load_masks(npz_path: Path, threshold: float) -> dict[str, NDArray[np.bool_]]:
    data = np.load(npz_path)
    prediction = np.asarray(data["prediction"]).squeeze() >= threshold
    target = np.asarray(data["target"]).squeeze() >= threshold
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must be matching 3D volumes")
    return {"prediction": prediction, "target": target}


def _classified_surface_points(
    prediction: NDArray[np.bool_], target: NDArray[np.bool_]
) -> dict[str, NDArray[np.float32]]:
    overlap = _surface_points(prediction & target)
    false_negative = _surface_points((~prediction) & target)
    false_positive = _surface_points(prediction & (~target))
    return {
        "Overlap: prediction and ground truth": overlap,
        "False negatives: missed target": false_negative,
        "False positives: extra prediction": false_positive,
    }


def _surface_points(mask: NDArray[np.bool_]) -> NDArray[np.float32]:
    if not mask.any():
        return np.zeros((0, 3), dtype=np.float32)
    padded = np.pad(mask, 1, constant_values=False)
    center = padded[1:-1, 1:-1, 1:-1]
    interior = (
        padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return np.argwhere(center & ~interior).astype(np.float32)


def _draw_scene(ax, points: dict[str, NDArray[np.float32]], *, azim: float) -> None:
    colors = {
        "Overlap: prediction and ground truth": ("#f2c94c", 10, 0.85),
        "False negatives: missed target": ("#2d9cdb", 18, 1.0),
        "False positives: extra prediction": ("#eb5757", 18, 1.0),
    }
    for label, xyz in points.items():
        if len(xyz) == 0:
            continue
        color, size, alpha = colors[label]
        ax.scatter(xyz[:, 2], xyz[:, 1], xyz[:, 0], s=size, c=color, alpha=alpha, label=label)
    ax.view_init(elev=22, azim=azim)
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = _axis_bounds(points)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.set_axis_off()
    ax.set_title("3D Reconstruction: yellow overlap, blue missed target, red extra prediction")
    ax.legend(loc="upper left", fontsize=8)


def _axis_bounds(points: dict[str, NDArray[np.float32]]) -> tuple[tuple[float, float], ...]:
    nonempty = [xyz for xyz in points.values() if len(xyz)]
    if not nonempty:
        return (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
    xyz = np.concatenate(nonempty, axis=0)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(max(maxs - mins) / 2.0 + 3.0)
    z_bounds = (float(center[0] - radius), float(center[0] + radius))
    y_bounds = (float(center[1] - radius), float(center[1] + radius))
    x_bounds = (float(center[2] - radius), float(center[2] + radius))
    return x_bounds, y_bounds, z_bounds


def _plotly_trace(go, xyz: NDArray[np.float32], name: str, color: str, size: int):
    return go.Scatter3d(
        x=xyz[:, 2] if len(xyz) else [],
        y=xyz[:, 1] if len(xyz) else [],
        z=xyz[:, 0] if len(xyz) else [],
        mode="markers",
        marker=dict(size=size, color=color, opacity=0.85),
        name=name,
    )

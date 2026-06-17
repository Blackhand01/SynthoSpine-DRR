"""Marching-cubes mesh export and visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def export_mesh_assets(npz_path: Path, output_dir: Path, *, level: float = 0.5, sigma: float = 0.7) -> dict:
    """Export smooth prediction/target meshes, HTML viewer, and GIF rotation."""

    volumes = _load_volumes(npz_path)
    pred_mesh = _build_mesh(volumes["prediction"], level=level, sigma=sigma)
    target_mesh = _build_mesh(volumes["target"], level=level, sigma=sigma)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "12_prediction_mesh_smooth.ply"
    target_path = output_dir / "13_target_mesh_smooth.ply"
    html_path = output_dir / "14_mesh_comparison_interactive.html"
    gif_path = output_dir / "15_mesh_rotation.gif"
    _write_ply(pred_mesh, pred_path, color=(235, 151, 45))
    _write_ply(target_mesh, target_path, color=(45, 156, 219))
    _write_html(pred_mesh, target_mesh, html_path)
    _write_gif(pred_mesh, target_mesh, gif_path)
    return {
        "prediction_vertices": int(len(pred_mesh["vertices"])),
        "prediction_faces": int(len(pred_mesh["faces"])),
        "target_vertices": int(len(target_mesh["vertices"])),
        "target_faces": int(len(target_mesh["faces"])),
        "prediction_ply": str(pred_path),
        "target_ply": str(target_path),
        "html": str(html_path),
        "gif": str(gif_path),
    }


def _load_volumes(npz_path: Path) -> dict[str, NDArray[np.float32]]:
    data = np.load(npz_path)
    return {
        "prediction": np.asarray(data["prediction"]).squeeze().astype(np.float32),
        "target": np.asarray(data["target"]).squeeze().astype(np.float32),
    }


def _build_mesh(volume: NDArray[np.float32], *, level: float, sigma: float) -> dict[str, NDArray]:
    try:
        from scipy.ndimage import gaussian_filter
        from skimage import measure
    except ImportError as exc:
        raise RuntimeError("scipy and scikit-image are required for smooth mesh export") from exc
    mask = (volume >= level).astype(np.float32)
    if not mask.any():
        raise ValueError("Cannot build a mesh from an empty thresholded volume")
    smoothed = gaussian_filter(mask, sigma=sigma) if sigma > 0 else mask
    threshold = min(0.5, float(smoothed.max()) * 0.75)
    vertices, faces, normals, values = measure.marching_cubes(smoothed, level=threshold)
    vertices = _laplacian_smooth(vertices.astype(np.float32), faces.astype(np.int32))
    return {
        "vertices": vertices,
        "faces": faces.astype(np.int32),
        "normals": normals.astype(np.float32),
        "values": values.astype(np.float32),
    }


def _laplacian_smooth(vertices: NDArray[np.float32], faces: NDArray[np.int32]) -> NDArray[np.float32]:
    neighbors = [set() for _ in range(len(vertices))]
    for a, b, c in faces:
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    current = vertices.copy()
    for _ in range(4):
        updated = current.copy()
        for index, linked in enumerate(neighbors):
            if linked:
                updated[index] = current[index] * 0.65 + current[list(linked)].mean(axis=0) * 0.35
        current = updated
    return current


def _write_ply(mesh: dict[str, NDArray], path: Path, *, color: tuple[int, int, int]) -> None:
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {len(faces)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for z, y, x in vertices:
            handle.write(f"{x:.4f} {y:.4f} {z:.4f} {color[0]} {color[1]} {color[2]}\n")
        for a, b, c in faces:
            handle.write(f"3 {a} {b} {c}\n")


def _write_html(pred_mesh: dict[str, NDArray], target_mesh: dict[str, NDArray], path: Path) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("plotly is required for interactive mesh export") from exc
    fig = go.Figure(data=[_mesh_trace(go, target_mesh, "Ground truth", "#2d9cdb", 0.35), _mesh_trace(go, pred_mesh, "Prediction", "#eb9745", 0.65)])
    fig.update_layout(title="Smooth 3D Mesh Comparison", scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=45, b=0))
    fig.write_html(path, include_plotlyjs="cdn")


def _write_gif(pred_mesh: dict[str, NDArray], target_mesh: dict[str, NDArray], path: Path) -> None:
    try:
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib and imageio are required for mesh GIF export") from exc
    frames = []
    for index in range(48):
        fig = plt.figure(figsize=(8, 6), dpi=130)
        ax = fig.add_subplot(111, projection="3d")
        _plot_mesh(ax, target_mesh, color="#2d9cdb", alpha=0.28)
        _plot_mesh(ax, pred_mesh, color="#eb9745", alpha=0.72)
        _frame_axes(ax, [pred_mesh["vertices"], target_mesh["vertices"]], 360 * index / 48)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(path, frames, fps=10)


def _mesh_trace(go, mesh: dict[str, NDArray], name: str, color: str, opacity: float):
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    return go.Mesh3d(
        x=vertices[:, 2], y=vertices[:, 1], z=vertices[:, 0],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        name=name, color=color, opacity=opacity, flatshading=False,
    )


def _plot_mesh(ax, mesh: dict[str, NDArray], *, color: str, alpha: float) -> None:
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    ax.plot_trisurf(vertices[:, 2], vertices[:, 1], vertices[:, 0], triangles=faces, color=color, alpha=alpha, linewidth=0.0, shade=True)


def _frame_axes(ax, vertex_sets: list[NDArray[np.float32]], azim: float) -> None:
    vertices = np.concatenate(vertex_sets, axis=0)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(max(maxs - mins) / 2.0 + 2.0)
    ax.set_xlim(center[2] - radius, center[2] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[0] - radius, center[0] + radius)
    ax.view_init(elev=22, azim=azim)
    ax.set_axis_off()
    ax.set_title("Smooth mesh: orange prediction, blue ground truth")

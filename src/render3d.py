"""Command-line export for interactive and rotating 3D reconstruction views."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation.render3d import export_interactive_html, export_rotation_gif


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 3D prediction/target comparison assets.")
    parser.add_argument("--npz", type=Path, default=Path("runs/view_overfit_real/volumes/prediction_target.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/reconstruction_assets"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--frames", type=int, default=48)
    args = parser.parse_args()

    gif_path = args.output_dir / "10_reconstruction_rotation.gif"
    html_path = args.output_dir / "11_reconstruction_interactive.html"
    export_rotation_gif(args.npz, gif_path, threshold=args.threshold, frames=args.frames)
    export_interactive_html(args.npz, html_path, threshold=args.threshold)
    print(f"Wrote {gif_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()

"""Command-line export for smooth marching-cubes reconstruction meshes."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation.mesh import export_mesh_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Export smooth 3D reconstruction mesh assets.")
    parser.add_argument("--npz", type=Path, default=Path("runs/view_overfit_real/volumes/prediction_target.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/recruiter_assets"))
    parser.add_argument("--level", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.7)
    args = parser.parse_args()
    summary = export_mesh_assets(args.npz, args.output_dir, level=args.level, sigma=args.sigma)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

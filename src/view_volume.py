"""Export 3D reconstruction artifacts for external viewers."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Export 3D prediction/target volumes and PLY surfaces.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/view_volume"))
    args = parser.parse_args()
    summary = run_evaluation(
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        manifest_path=args.manifest,
        sample_index=args.sample_index,
    )
    print(f"Wrote 3D artifacts to {args.output_dir}")
    print(f"F1={summary['metrics']['f1']:.4f} IoU={summary['metrics']['iou']:.4f}")


if __name__ == "__main__":
    main()


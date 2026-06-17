"""Create patient-level train/validation manifest splits."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .manifest import read_manifest, write_manifest


def split_manifest(
    manifest_path: str | Path,
    train_output: str | Path,
    val_output: str | Path,
    *,
    val_fraction: float = 0.2,
    seed: int = 23,
) -> dict[str, int]:
    """Split records by patient_id and write train/val JSONL manifests."""

    records = read_manifest(manifest_path)
    patients = sorted({record["patient_id"] for record in records})
    rng = random.Random(seed)
    rng.shuffle(patients)
    val_count = max(1, round(len(patients) * val_fraction)) if len(patients) > 1 else 0
    val_patients = set(patients[:val_count])
    train_records = [record for record in records if record["patient_id"] not in val_patients]
    val_records = [record for record in records if record["patient_id"] in val_patients]
    if not train_records or not val_records:
        raise ValueError("Split requires at least one train and one validation record")
    write_manifest(_as_records(train_records), train_output)
    write_manifest(_as_records(val_records), val_output)
    return {"train": len(train_records), "val": len(val_records)}


def _as_records(records: list[dict]):
    from ..core.schemas import ManifestRecord

    return [
        ManifestRecord(
            record["sample_id"],
            record["patient_id"],
            record["anatomy_id"],
            record["ct_path"],
            record["mask_path"],
            record["sample_path"],
            tuple(record["spacing_mm"]),
            tuple(record["views"]),
        )
        for record in records
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Split manifest by patient_id.")
    parser.add_argument("--manifest", default="data/processed/real_split/manifest.jsonl")
    parser.add_argument("--train-output", default="data/processed/real_split/manifest_train.jsonl")
    parser.add_argument("--val-output", default="data/processed/real_split/manifest_val.jsonl")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    counts = split_manifest(args.manifest, args.train_output, args.val_output, val_fraction=args.val_fraction, seed=args.seed)
    print(f"Wrote train={counts['train']} val={counts['val']}")


if __name__ == "__main__":
    main()

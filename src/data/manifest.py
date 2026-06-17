"""Manifest building and JSONL IO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.config_io import load_config
from ..core.schemas import ManifestRecord
from .sampling import discover_patient_pairs, sample_patients


DEFAULTS = {
    "data_root": "data/raw/ctspine1k",
    "processed_root": "data/processed/micro_ctspine",
    "max_patients": 20,
    "seed": 23,
    "views": ("ap", "lateral", "oblique", "misc"),
}


def build_manifest(config: dict) -> list[ManifestRecord]:
    """Create manifest records from discovered patient CT/mask pairs."""

    processed = Path(config["processed_root"])
    pairs = sample_patients(
        discover_patient_pairs(config["data_root"]),
        max_patients=int(config["max_patients"]),
        seed=int(config["seed"]),
    )
    records = []
    for patient_id, ct_path, mask_path in pairs:
        sample_path = processed / "samples" / f"{patient_id}_L3.npz"
        records.append(
            ManifestRecord(
                f"{patient_id}_L3",
                patient_id,
                "L3",
                str(ct_path),
                str(mask_path),
                str(sample_path),
                (1.0, 1.0, 1.0),
                tuple(config["views"]),
            )
        )
    return records


def write_manifest(records: list[ManifestRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.__dict__) + "\n")


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CTSpine micro manifest.")
    parser.add_argument("--config", default="configs/micro_ctspine.yaml")
    parser.add_argument("--output", default="data/processed/micro_ctspine/manifest.jsonl")
    args = parser.parse_args()
    config = load_config(args.config, DEFAULTS)
    records = build_manifest(config)
    write_manifest(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()


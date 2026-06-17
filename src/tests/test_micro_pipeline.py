"""Tests for the CTSpine micro-pipeline PoC."""

from __future__ import annotations

import numpy as np

from ..data.manifest import build_manifest, write_manifest
from ..data.sampling import sample_patients
from ..drr.generator import generate_record_sample
from ..evaluation.surface import surface_metrics
from ..training.subset import run_subset_training


def test_patient_sampler_seed_and_limit(tmp_path) -> None:
    pairs = [(f"p{i}", tmp_path / f"p{i}.npz", tmp_path / f"p{i}_mask.npz") for i in range(30)]
    first = sample_patients(pairs, max_patients=10, seed=7)
    second = sample_patients(pairs, max_patients=10, seed=7)
    assert first == second
    assert len(first) == 10


def test_manifest_builder_fields(tmp_path) -> None:
    _write_pair(tmp_path, "patient001")
    records = build_manifest(
        {
            "data_root": str(tmp_path),
            "processed_root": str(tmp_path / "processed"),
            "max_patients": 20,
            "seed": 23,
            "views": ("ap", "lateral", "oblique", "misc"),
        }
    )
    assert len(records) == 1
    record = records[0]
    assert record.ct_path.endswith("patient001.npz")
    assert record.mask_path.endswith("patient001_mask.npz")
    assert record.views == ("ap", "lateral", "oblique", "misc")


def test_generate_one_drr_proxy_sample(tmp_path) -> None:
    ct_path, mask_path = _write_pair(tmp_path, "patient002")
    record = {
        "sample_id": "patient002_L3",
        "patient_id": "patient002",
        "anatomy_id": "L3",
        "ct_path": str(ct_path),
        "mask_path": str(mask_path),
        "sample_path": str(tmp_path / "sample.npz"),
        "spacing_mm": (1.0, 1.0, 1.0),
        "views": ("ap", "lateral", "oblique", "misc"),
    }
    result = generate_record_sample(record, crop_size=32, volume_size=16)
    data = np.load(result["sample_path"])
    assert data["images"].shape == (4, 1, 32, 32)
    assert data["projections"].shape == (4, 3, 4)
    assert data["target"].shape == (1, 16, 16, 16)
    assert int(data["vertebra_label"]) == 22


def test_subset_training_one_step(tmp_path) -> None:
    _write_pair(tmp_path, "patient003")
    records = build_manifest(
        {
            "data_root": str(tmp_path),
            "processed_root": str(tmp_path / "processed"),
            "max_patients": 1,
            "seed": 1,
            "views": ("ap", "lateral", "oblique", "misc"),
        }
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(records, manifest)
    generate_record_sample(records[0].__dict__, crop_size=32, volume_size=16)
    final = run_subset_training(
        {
            "manifest_path": str(manifest),
            "output_dir": str(tmp_path / "run"),
            "volume_size": 16,
            "epochs": 1,
            "learning_rate": 0.001,
        }
    )
    assert (tmp_path / "run" / "best.pt").exists()
    assert final["loss"] > 0.0


def test_surface_identity_ckdtree_path() -> None:
    volume = np.zeros((1, 12, 12, 12), dtype=np.float32)
    volume[:, 3:9, 3:9, 3:9] = 1.0
    metrics = surface_metrics(volume, volume, tolerance_mm=0.1)
    assert metrics.surface_score > 0.999
    assert metrics.asd_mm == 0.0


def _write_pair(root, patient_id: str):
    volume = np.zeros((12, 12, 12), dtype=np.float32)
    mask = np.zeros_like(volume)
    volume[3:9, 3:9, 3:9] = 120.0
    mask[3:9, 3:9, 3:9] = 22.0
    ct_path = root / f"{patient_id}.npz"
    mask_path = root / f"{patient_id}_mask.npz"
    np.savez_compressed(ct_path, volume=volume, spacing_mm=np.array([1.0, 1.0, 1.0]))
    np.savez_compressed(mask_path, mask=mask)
    return ct_path, mask_path

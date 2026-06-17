"""Patient sampling utilities for CTSpine-style datasets."""

from __future__ import annotations

import random
from pathlib import Path


CT_EXTENSIONS = (".nii", ".nii.gz", ".mha", ".mhd", ".npz")


def discover_patient_pairs(data_root: str | Path) -> list[tuple[str, Path, Path]]:
    """Discover CT/mask pairs using common filename conventions."""

    root = Path(data_root)
    candidates = []
    for ct_path in _iter_medical_files(root):
        name = ct_path.name.lower()
        if any(token in name for token in ("mask", "seg", "label")):
            continue
        mask_path = _find_mask_for_ct(ct_path)
        if mask_path is not None:
            candidates.append((ct_path.stem.replace(".nii", ""), ct_path, mask_path))
    return sorted(candidates, key=lambda item: item[0])


def sample_patients(
    pairs: list[tuple[str, Path, Path]],
    *,
    max_patients: int = 20,
    seed: int = 23,
) -> list[tuple[str, Path, Path]]:
    """Deterministically sample up to max_patients patient pairs."""

    rng = random.Random(seed)
    pairs_copy = list(pairs)
    rng.shuffle(pairs_copy)
    return sorted(pairs_copy[:max_patients], key=lambda item: item[0])


def _iter_medical_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if path.is_file() and any(str(path).endswith(ext) for ext in CT_EXTENSIONS):
            files.append(path)
    return files


def _find_mask_for_ct(ct_path: Path) -> Path | None:
    parallel = _find_parallel_hf_label(ct_path)
    if parallel is not None:
        return parallel
    stem = ct_path.name
    replacements = [
        stem.replace("image", "mask"),
        stem.replace("ct", "mask"),
        stem.replace("volume", "mask"),
        stem.replace(".nii.gz", "_mask.nii.gz"),
        stem.replace(".npz", "_mask.npz"),
    ]
    for name in replacements:
        candidate = ct_path.with_name(name)
        if candidate.exists() and candidate != ct_path:
            return candidate
    for token in ("mask", "seg", "label"):
        matches = list(ct_path.parent.glob(f"*{ct_path.stem}*{token}*"))
        if matches:
            return matches[0]
    return None


def _find_parallel_hf_label(ct_path: Path) -> Path | None:
    parts = list(ct_path.parts)
    if "volumes" not in parts:
        return None
    idx = parts.index("volumes")
    parts[idx] = "labels"
    label_name = ct_path.name
    for suffix in (".nii.gz", ".nii"):
        if label_name.endswith(suffix):
            label_name = label_name[: -len(suffix)] + "_seg" + suffix
            break
    parts[-1] = label_name
    candidate = Path(*parts)
    return candidate if candidate.exists() else None

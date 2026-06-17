"""Download a CTSpine1K micro-subset from Hugging Face."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


REPO_ID = "alexanderdann/CTSpine1K"


def download_micro_subset(output_dir: str | Path, max_pairs: int = 20, retries: int = 3) -> list[tuple[str, str]]:
    """Download up to max_pairs raw CT/mask pairs into output_dir."""

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub: pip install huggingface_hub") from exc

    files = _with_retries(lambda: HfApi().list_repo_files(REPO_ID, repo_type="dataset"), retries)
    pairs = _pair_files(files)[:max_pairs]
    if not pairs:
        raise RuntimeError("No CT/mask pairs found in Hugging Face file listing")
    output = Path(output_dir)
    for ct_file, mask_file in pairs:
        for filename in (ct_file, mask_file):
            _with_retries(
                lambda name=filename: hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=name,
                    local_dir=output,
                ),
                retries,
            )
    return pairs


def _pair_files(files: list[str]) -> list[tuple[str, str]]:
    volumes = [path for path in files if "/volumes/" in path and path.endswith((".nii.gz", ".nii"))]
    labels = [path for path in files if "/labels/" in path and path.endswith((".nii.gz", ".nii"))]
    label_map = {_case_key(path).replace("_seg", ""): path for path in labels}
    pairs = []
    for volume in sorted(volumes):
        label = label_map.get(_case_key(volume))
        if label is not None:
            pairs.append((volume, label))
    return pairs


def _case_key(path: str) -> str:
    name = Path(path).name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _with_retries(fn, retries: int):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # network libraries raise several transient types
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CTSpine1K micro-subset from Hugging Face.")
    parser.add_argument("--output-dir", default="data/raw/ctspine1k")
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    pairs = download_micro_subset(args.output_dir, args.max_pairs, args.retries)
    print(f"Downloaded {len(pairs)} CT/mask pairs to {args.output_dir}")


if __name__ == "__main__":
    main()


# Spine2Space

Geometry-aware sparse X-ray to 3D reconstruction MVP inspired by X23D.

This repository is a recruiter-facing technical prototype. It does not claim clinical performance. It demonstrates the engineering spine of an X23D-like system:

- explicit projection matrices;
- crop-adjusted calibration metadata;
- synthetic multi-view X-ray/DRR placeholders;
- 2D encoder, differentiable back-projection, multi-view fusion, 3D refiner;
- voxel and surface-style evaluation artifacts.

## Run Locally

Optional visualization dependencies for mesh export:

```bash
python -m pip install -e ".[visualization]"
```

```bash
python -m src.test_smoke
python -m src.cli --volume-size 16
python -m src.evaluate --volume-size 16 --output-dir runs/demo_eval
```

## Micro CTSpine PoC

```bash
python -m src.data.download_ctspine --output-dir data/raw/ctspine1k --max-pairs 20
python -m src.data.manifest --config configs/micro_ctspine.yaml
python -m src.drr.generator --config configs/micro_ctspine.yaml --limit 2
python -m src.train_overfit --config configs/overfit.yaml
python -m src.train_subset --config configs/subset_train.yaml
python -m src.evaluate --volume-size 16 --checkpoint runs/overfit_synthetic/best.pt --output-dir runs/eval_overfit
```

## 3D Visualization

```bash
python -m src.view_volume \
  --checkpoint runs/overfit_real/best.pt \
  --manifest data/processed/micro_ctspine/manifest.jsonl \
  --sample-index 0 \
  --output-dir runs/view_overfit_real
```

Open the generated PLY files in MeshLab, Blender, CloudCompare, or Open3D:

- `runs/view_overfit_real/meshes/prediction_surface.ply`
- `runs/view_overfit_real/meshes/target_surface.ply`

The raw voxel probabilities and target are saved in:

- `runs/view_overfit_real/volumes/prediction_target.npz`

Generate recruiter-facing 3D video and browser viewer:

```bash
python -m src.render3d \
  --npz runs/view_overfit_real/volumes/prediction_target.npz \
  --output-dir reports/recruiter_assets
```

Generate smooth marching-cubes meshes:

```bash
python -m src.mesh3d \
  --npz runs/view_overfit_real/volumes/prediction_target.npz \
  --output-dir reports/recruiter_assets
```

## Key Outputs

Evaluation writes:

- `metrics.json`
- `threshold_sweep.csv`
- `report.md`
- `qualitative/overlay_axial.png`
- `qualitative/target_slices.png`
- `qualitative/prediction_slices.png`

Recruiter-facing report and assets:

- `reports/recruiter_report.md`
- `reports/recruiter_assets/`

## Source Layout

- `src/core/`: configuration and typed schemas.
- `src/geometry/`: projection matrices and crop-adjusted geometry.
- `src/data/`: preprocessing, batching, and synthetic sample generation.
- `src/models/`: PyTorch layers and X23D-like baseline.
- `src/evaluation/`: voxel/surface metrics, reporting, visualization, evaluation runner.
- `src/apps/`: command-line application implementations.
- `src/cli.py` and `src/evaluate.py`: stable wrapper entrypoints.
- `src/tests/`: pytest smoke and evaluation tests.

## Current Scope

Implemented:

- data schemas and batching;
- projection utilities and crop adjustment;
- CTSpine1K micro-dataset download, manifest generation, and CT-derived DRR proxy samples;
- L3 label extraction from CTSpine masks instead of full nonzero-mask targets;
- synthetic and real-sample overfit loops;
- 20-sample subset training loop;
- X23D-like PyTorch baseline;
- F1, IoU, cKDTree surface metrics, PNG qualitative artifacts, GIF, point-cloud PLY, and smooth marching-cubes mesh exports.

Next technical steps:

- more physical DRR generation;
- patient-level train/validation/test split;
- GPU training with larger model capacity;
- calibration sensitivity analysis;
- patient-level generalization beyond one-sample overfit.

The PoC now uses cKDTree surface distances when SciPy is available.

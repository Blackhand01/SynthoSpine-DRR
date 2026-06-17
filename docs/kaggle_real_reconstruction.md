# Kaggle Real Reconstruction Run

## Purpose

This run is the first non-overfit reconstruction check:

```text
train patients -> held-out validation patients
```

It is expected to be difficult. A low validation F1 is not a failure of the pipeline; it is evidence that we are finally measuring generalization instead of memorization.

## Notebook

Use:

```text
notebooks/kaggle_real_reconstruction.ipynb
```

The notebook has six numbered cells:

1. setup project and dependencies;
2. download CTSpine1K micro-subset and generate CT-derived DRR samples;
3. create patient-level train/validation manifests;
4. train with validation tracking;
5. evaluate the best checkpoint on a held-out validation sample;
6. package outputs for download.

## Kaggle Setup

Recommended settings:

```text
Accelerator: GPU T4 or P100
Internet: on
Dataset size: start with 20 CT/mask pairs
Batch size: 1
Grid: 64 x 64 x 64
DRR crop: 128 x 128
```

The project must be available at:

```text
/kaggle/working/Spine2Space
```

If you upload it elsewhere, edit `PROJECT_DIR` in cell 01.

## Expected Outputs

Download:

```text
spine2space_real_split_outputs.tar.gz
```

Inside:

```text
runs/real_split_train/
runs/real_split_eval/
reports/real_split_assets/
```

Key files:

```text
runs/real_split_train/best.pt
runs/real_split_train/metrics.json
runs/real_split_eval/metrics.json
reports/real_split_assets/14_mesh_comparison_interactive.html
reports/real_split_assets/15_mesh_rotation.mp4
```

## What Good Looks Like

Minimum useful result:

```text
training completes
validation metrics are exported
held-out prediction mesh exists
failure case is inspectable
```

Good PoC result:

```text
val_f1 > 0.10
val_iou > 0.05
prediction is localized near the target vertebra
```

Suspicious result:

```text
val_f1 > 0.80 immediately
```

That would likely indicate data leakage or a target/input contract bug.

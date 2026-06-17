# Colab/Kaggle Micro-Pipeline Notes

## Expected Layout

Mount or copy CTSpine-style files into:

```text
data/raw/ctspine1k/
```

Accepted local formats:

- `.npz` fixtures with `volume` and `spacing_mm`;
- matching mask `.npz` files with `mask`;
- SimpleITK-readable CT/mask files when `SimpleITK` is installed.

## Recommended Kaggle Order

```bash
pip install -e ".[model,medical,plots]"
python -m src.test_smoke
python -m src.data.manifest --config configs/micro_ctspine.yaml
python -m src.drr.generator --config configs/micro_ctspine.yaml --limit 20
python -m src.train_overfit --config configs/overfit.yaml
python -m src.train_subset --config configs/subset_train.yaml
```

## Scope

This is a PoC pipeline correctness run. It is not a clinical validation run.


# SimNorth

[![DOI](https://zenodo.org/badge/1256404282.svg)](https://doi.org/10.5281/zenodo.20497228)

Contrastive self-supervised representation learning for ultrasound frames.

SimNorth maps frames onto the unit hypersphere with three objectives:

1. **Alignment** — two augmented views of the same frame are pulled together.
2. **Uniformity** — randomly paired views are pushed apart with a rank-based
   weighting, so the most-similar (likely true-negative) random pairs are
   penalized the most.
3. **North / light house** — a set of fixed anchor points ("light houses") on
   the sphere attract their nearest embeddings, organizing the learned manifold
   around well-separated anchors.

Built on **PyTorch Lightning 2.x** with **MLflow** experiment tracking.

## Install

```bash
pip install -e .
# or
pip install -r requirements.txt
```

## Data format

A CSV or Parquet table with one row per frame and a column of image paths
(default `img_path`), resolved relative to `--mount_point`. Frames are read with
SimpleITK; single-channel frames are promoted to 3 channels.

## Usage

Optionally pre-compute well-separated light house anchors:

```bash
python pretrain_lighthouse.py --n_lights 64 --emb_dim 128 --out lights.pkl
```

Train:

```bash
python train.py \
    --csv_train train.parquet --csv_valid valid.parquet --csv_test test.parquet \
    --mount_point /data/frames \
    --base_encoder efficientnet_b0 --emb_dim 128 --hidden_dim 64 --n_lights 64 \
    --batch_size 256 --epochs 200 --lr 1e-4 \
    --lights lights.pkl \
    --tracking_uri file:./mlruns --experiment_name SimNorth --run_name effnet_b0
```

If `--lights` is omitted, anchors are initialized uniformly at random.

Inspect runs:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Package layout

```
simnorth/
├── nets/
│   ├── simnorth.py    # SimNorth, ProjectionHead, GaussianNoise
│   └── lighthouse.py  # LightHouse anchor optimizer
├── data/
│   ├── dataset.py     # USDataset, USDataModule
│   └── transforms.py  # paired augmentation transforms
└── callbacks/
    └── image_logger.py  # logs augmented-view grids to MLflow
train.py                  # training entrypoint (MLflow)
pretrain_lighthouse.py    # optional anchor pre-optimization
```

## Provenance

Extracted and modernized from the `us-famli-pl` research codebase
(`nets/contrastive.py`, `contrastive_learning.py`). Compared to the original:

- Upgraded from `pytorch_lightning` to the `lightning` 2.x API.
- Switched experiment logging from Neptune to MLflow.
- `SimNorth` takes explicit hyperparameters instead of an `args` namespace.
- Fixed a copy-paste bug in the north loss where the second view's nearest
  index was taken from the first view (`z_1_n` now uses the `z_1` index).
```

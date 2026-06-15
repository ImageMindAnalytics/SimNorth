# SimNorth

[![DOI](https://zenodo.org/badge/1256404282.svg)](https://doi.org/10.5281/zenodo.20497228)

Contrastive self-supervised representation learning for ultrasound frames.

SimNorth maps frames onto the unit hypersphere with three objectives:

1. **Alignment** — two augmented views of the same frame are pulled together.
2. **Uniformity** — randomly paired views are pushed apart with a rank-based
   weighting, so the most-similar random pairs (likely false negatives) are
   penalized the least, while the most-different pairs are penalized the most.
3. **Prototypical (ProtoNCE)** — embeddings are pulled toward data-driven
   cluster centroids, re-estimated each epoch by KMeans (the
   Expectation-Maximization scheme of [Prototypical Contrastive Learning](https://arxiv.org/abs/2005.04966),
   Li et al., ICLR 2021). The prototype term is bootstrapped by `--warmup_epochs`
   of alignment + uniformity only. See `simnorth/docs/protonce_design.md`.

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

Train:

```bash
python train.py \
    --nn SimNorth --data_module USDataModule \
    --csv_train train.parquet --csv_valid valid.parquet --csv_test test.parquet \
    --mount_point /data/frames \
    --base_encoder efficientnet_b0 --emb_dim 128 --hidden_dim 64 \
    --batch_size 256 --epochs 200 --lr 1e-4 \
    --warmup_epochs 20 --proto_tau 0.1 --proto_weight 1.0 \
    --tracking_uri file:./mlruns --experiment_name SimNorth --run_name effnet_b0
```

`--nn` / `--data_module` select the network and data-module classes by name from
the `simnorth` package; each contributes its own CLI arguments via
`add_model_specific_args` / `add_data_specific_args` (run `python train.py --help`
to see them). Both default to `SimNorth` / `USDataModule`, so the flags above are
optional.

The ProtoNCE term clusters the validation embeddings each epoch after
`--warmup_epochs`. By default the cluster counts `K` are derived from the
silhouette-optimal `k*`; pass `--proto_clusters "8,16,32"` to set them explicitly.

Inspect runs:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Package layout

```
simnorth/
├── nets/
│   └── simnorth.py    # SimNorth, ProjectionHead, GaussianNoise
├── data/
│   ├── dataset.py     # USDataset, USDataModule
│   └── transforms.py  # paired augmentation transforms
├── callbacks/
│   └── image_logger.py  # logs augmented-view grids to MLflow
└── docs/
    └── protonce_design.md  # ProtoNCE design doc
train.py                  # training entrypoint (MLflow)
```

## Provenance

Extracted and modernized from the `us-famli-pl` research codebase
(`nets/contrastive.py`, `contrastive_learning.py`). Compared to the original:

- Upgraded from `pytorch_lightning` to the `lightning` 2.x API.
- Switched experiment logging from Neptune to MLflow.
- `SimNorth` takes explicit hyperparameters instead of an `args` namespace.
- Replaced the fixed "light house" anchor term with a data-driven ProtoNCE
  prototype term (see `simnorth/docs/protonce_design.md`).
```

"""Train SimNorth contrastive self-supervised model.

Logs to MLflow. Example:

    python train.py \
        --csv_train train.parquet --csv_valid valid.parquet --csv_test test.parquet \
        --mount_point /data/frames \
        --base_encoder efficientnet_b0 --emb_dim 128 --n_lights 64 \
        --batch_size 256 --epochs 200 \
        --tracking_uri file:./mlruns --experiment_name SimNorth --run_name effnet_b0
"""

import argparse
import os
import pickle

import pandas as pd
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import MLFlowLogger

from simnorth import (
    SimNorth,
    USDataModule,
    SimTrainTransforms,
    SimTrainTransformsV2,
    SimEvalTransforms,
    SimNorthImageLogger,
)


def _read_table(path):
    if os.path.splitext(path)[1] == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def main(args):
    df_train = _read_table(os.path.join(args.mount_point, args.csv_train))
    df_val = _read_table(os.path.join(args.mount_point, args.csv_valid))
    df_test = _read_table(os.path.join(args.mount_point, args.csv_test)) if args.csv_test else None

    if args.query:
        df_train = df_train.query(args.query).reset_index(drop=True)
        df_val = df_val.query(args.query).reset_index(drop=True)
        if df_test is not None:
            df_test = df_test.query(args.query).reset_index(drop=True)

    if args.train_transform == 2:
        print("Using SimTrainTransformsV2")
        train_transform = SimTrainTransformsV2(args.img_size)
    else:
        train_transform = SimTrainTransforms(args.img_size)
    valid_transform = SimEvalTransforms(args.img_size)

    lights = None
    if args.lights is not None:
        with open(args.lights, "rb") as f:
            lights = pickle.load(f)

    model = SimNorth(
        base_encoder=args.base_encoder,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        n_lights=args.n_lights,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        w=args.w,
        light_house=lights,
    )

    datamodule = USDataModule(
        df_train,
        df_val,
        df_test,
        mount_point=args.mount_point,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_column=args.img_column,
        train_transform=train_transform,
        valid_transform=valid_transform,
        drop_last=True,
    )

    logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        tracking_uri=args.tracking_uri,
        log_model=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename="{epoch}-{val_loss:.2f}",
        save_top_k=2,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.0, patience=args.patience, verbose=True, mode="min"
    )
    image_logger = SimNorthImageLogger(log_steps=args.log_steps)

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=[early_stop_callback, checkpoint_callback, image_logger],
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.model)


def get_argparse():
    parser = argparse.ArgumentParser(description="Train SimNorth contrastive model")

    # Data
    parser.add_argument("--csv_train", required=True, type=str, help="Train CSV/parquet")
    parser.add_argument("--csv_valid", required=True, type=str, help="Validation CSV/parquet")
    parser.add_argument("--csv_test", default=None, type=str, help="Test CSV/parquet (optional)")
    parser.add_argument("--mount_point", default="./", type=str, help="Dataset mount directory")
    parser.add_argument("--img_column", default="img_path", type=str, help="Image path column")
    parser.add_argument("--img_size", default=224, type=int, help="Square crop size")
    parser.add_argument("--query", default=None, type=str, help="Optional pandas query filter")
    parser.add_argument("--num_workers", default=4, type=int, help="Dataloader workers")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument("--train_transform", default=0, type=int, help="0=default, 2=V2 transforms")

    # Model
    parser.add_argument("--base_encoder", default="efficientnet_b0", type=str, help="torchvision encoder")
    parser.add_argument("--emb_dim", default=128, type=int, help="Embedding dimension")
    parser.add_argument("--hidden_dim", default=64, type=int, help="Projection head hidden dim")
    parser.add_argument("--n_lights", default=64, type=int, help="Number of light house anchors")
    parser.add_argument("--lights", default=None, type=str, help="Pickle file with light house anchors")
    parser.add_argument("--w", default=4.0, type=float, help="Weight scale for the contrastive (push) term")

    # Optimization
    parser.add_argument("--lr", "--learning-rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")
    parser.add_argument("--epochs", default=200, type=int, help="Max epochs")
    parser.add_argument("--steps", default=-1, type=int, help="Max steps (-1 = unlimited)")
    parser.add_argument("--patience", default=30, type=int, help="Early stopping patience")
    parser.add_argument("--model", default=None, type=str, help="Checkpoint to resume from")

    # Logging / output
    parser.add_argument("--out", default="./", type=str, help="Checkpoint output dir")
    parser.add_argument("--tracking_uri", default="file:./mlruns", type=str, help="MLflow tracking URI")
    parser.add_argument("--experiment_name", default="SimNorth", type=str, help="MLflow experiment name")
    parser.add_argument("--run_name", default=None, type=str, help="MLflow run name")
    parser.add_argument("--log_steps", default=100, type=int, help="Image logging interval (batches)")

    return parser


if __name__ == "__main__":
    main(get_argparse().parse_args())

"""Train SimNorth contrastive self-supervised model.

Dynamic dispatch on ``--nn`` / ``--data_module`` mirrors the FAMLI classification
trainer: the network and the data module each contribute their own CLI arguments
via ``add_model_specific_args`` / ``add_data_specific_args``, and are constructed
straight from the parsed namespace (``NN(**vars(args))`` / ``DM(**vars(args))``).

Logs to MLflow. Example:

    python train.py \
        --nn SimNorth --data_module USDataModule \
        --csv_train train.parquet --csv_valid valid.parquet --csv_test test.parquet \
        --mount_point /data/frames \
        --base_encoder efficientnet_b0 --emb_dim 128 --n_lights 64 \
        --batch_size 256 --epochs 200 \
        --tracking_uri file:./mlruns --experiment_name SimNorth --run_name effnet_b0
"""

import argparse
import json
import os

import torch

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import MLFlowLogger

import simnorth
from simnorth import SimNorthImageLogger, BestMetricTracker


def add_train_args(parser):
    """Add generic (network/data-module agnostic) training args. Returns the parser."""
    hparams_group = parser.add_argument_group("Hyperparameters")
    hparams_group.add_argument("--epochs", default=200, type=int, help="Max epochs")
    hparams_group.add_argument("--steps", default=-1, type=int, help="Max steps (-1 = unlimited)")
    hparams_group.add_argument("--patience", default=30, type=int, help="Early stopping patience")
    hparams_group.add_argument("--seed_everything", default=None, type=int, help="Seed for reproducibility")
    hparams_group.add_argument("--find_unused_parameters", default=0, type=int, help="DDP find_unused_parameters")
    hparams_group.add_argument("--accumulate_grad_batches", default=1, type=int, help="Accumulate gradient batches")

    input_group = parser.add_argument_group("Input")
    input_group.add_argument("--nn", default="SimNorth", type=str, help="Network class name in the simnorth package")
    input_group.add_argument("--data_module", default="USDataModule", type=str, help="Data module class name in the simnorth package")
    input_group.add_argument("--model", default=None, type=str, help="Checkpoint to resume from")

    output_group = parser.add_argument_group("Output")
    output_group.add_argument("--out", default="./", type=str, help="Checkpoint output dir")
    output_group.add_argument("--monitor", default="val_loss", type=str, help="Metric to monitor")
    output_group.add_argument("--monitor_mode", default="min", type=str, help="Monitor mode (min/max)")
    output_group.add_argument("--write_metric", default=None, type=str, help="Write best monitored metric to this JSON file (for Optuna subprocess)")

    log_group = parser.add_argument_group("Logging")
    log_group.add_argument("--tracking_uri", default="file:./mlruns", type=str, help="MLflow tracking URI")
    log_group.add_argument("--experiment_name", default="SimNorth", type=str, help="MLflow experiment name")
    log_group.add_argument("--run_name", default=None, type=str, help="MLflow run name")
    log_group.add_argument("--log_steps", default=5, type=int, help="Log scalars every N steps")
    log_group.add_argument("--image_log_steps", default=100, type=int, help="Log image grids every N steps")
    return parser


def main(args):
    if args.out and not os.path.exists(args.out):
        os.makedirs(args.out)

    deterministic = None
    if args.seed_everything:
        seed_everything(args.seed_everything, workers=True)
        deterministic = True

    NN = getattr(simnorth, args.nn)
    model = NN(**vars(args))

    DM = getattr(simnorth, args.data_module)
    datamodule = DM(**vars(args))

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename="{epoch}-{" + args.monitor + ":.2f}",
        save_top_k=2,
        monitor=args.monitor,
        mode=args.monitor_mode,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor=args.monitor, min_delta=0.0, patience=args.patience, verbose=True, mode=args.monitor_mode
    )
    image_logger = SimNorthImageLogger(log_steps=args.image_log_steps)
    best_tracker = BestMetricTracker(monitor=args.monitor, mode=args.monitor_mode)

    logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", args.tracking_uri),
        log_model=True,
    )

    trainer = Trainer(
        logger=logger,
        log_every_n_steps=args.log_steps,
        max_epochs=args.epochs,
        max_steps=args.steps,
        callbacks=[early_stop_callback, checkpoint_callback, image_logger, best_tracker],
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=args.find_unused_parameters)
        if torch.cuda.device_count() > 1
        else "auto",
        deterministic=deterministic,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.model)

    # Hand the best monitored metric back to the Optuna driver (rank 0 only).
    # best_tracker reads the monitor in on_validation_end, after it is logged.
    if args.write_metric and trainer.strategy.global_rank == 0:
        with open(args.write_metric, "w") as f:
            json.dump({best_tracker.monitor: best_tracker.best}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimNorth contrastive model", add_help=False)
    add_train_args(parser)

    args, _ = parser.parse_known_args()

    NN = getattr(simnorth, args.nn)
    NN.add_model_specific_args(parser)

    DM = getattr(simnorth, args.data_module)
    DM.add_data_specific_args(parser)

    parser = argparse.ArgumentParser(parents=[parser])
    args = parser.parse_args()

    main(args)

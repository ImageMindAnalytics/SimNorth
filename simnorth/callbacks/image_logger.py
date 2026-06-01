"""Lightning callback that logs a grid of augmented training views to MLflow."""

import matplotlib

matplotlib.use("Agg")  # headless backend for training nodes
import matplotlib.pyplot as plt

import torch
import torchvision

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger


class SimNorthImageLogger(Callback):
    def __init__(self, num_images=18, log_steps=100):
        self.num_images = num_images
        self.log_steps = log_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_steps != 0:
            return

        logger = trainer.logger
        if not isinstance(logger, MLFlowLogger):
            return

        img1, _ = batch
        max_num_image = min(img1.shape[0], self.num_images)
        grid = torchvision.utils.make_grid(img1[:max_num_image])

        fig = plt.figure(figsize=(7, 9))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")

        logger.experiment.log_figure(
            logger.run_id, fig, f"images/train_batch_{trainer.global_step}.png"
        )
        plt.close(fig)

"""LightHouse: optimize a set of anchor points to be maximally spread on the
positive orthant of the unit hypersphere.

The resulting anchors can be pickled and fed to :class:`SimNorth` via its
``light_house`` argument so training starts from well-separated anchors instead
of random ones.
"""

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader


class LightHouse(pl.LightningModule):
    def __init__(self, n_lights=64, emb_dim=128, lr=1e-3, momentum=0.9, weight_decay=1e-4, n_iter=1000):
        super().__init__()
        self.save_hyperparameters()

        self.light_house = nn.Parameter(
            torch.abs(F.normalize(torch.rand(n_lights, emb_dim)))
        )
        self.loss = nn.MSELoss()

    def configure_optimizers(self):
        return optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

    def compute_loss(self, mode):
        loss_north_c = 0
        for _ in range(self.hparams.n_lights - 1):
            r = torch.randperm(self.hparams.n_lights)
            light_house_r = self.light_house[r]
            loss_north_c = loss_north_c + self.loss(self.light_house, light_house_r)

        loss_north_c = torch.mean(loss_north_c)
        self.log(mode + "_loss", loss_north_c)
        # Maximize spread -> minimize the inverse of the mean pairwise distance.
        return 1.0 / (loss_north_c + 1e-7)

    def training_step(self, batch, batch_idx):
        # Keep anchors on the positive orthant of the unit sphere each step.
        self.light_house = nn.Parameter(torch.abs(F.normalize(self.light_house)))
        return self.compute_loss("train")

    def validation_step(self, batch, batch_idx):
        return self.compute_loss("val")

    def forward(self):
        return self.light_house

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.zeros(self.hparams.n_iter, 1)), batch_size=1)

    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)

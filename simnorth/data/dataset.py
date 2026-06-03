"""Dataset and LightningDataModule for SimNorth contrastive training.

The dataset reads ultrasound frames listed in a dataframe and applies a *pair*
transform (see :mod:`simnorth.data.transforms`). When a pair transform is used,
each sample is a ``(q, k)`` tuple and the default collate produces a batch
``(x_0, x_1)`` consumed by :class:`simnorth.nets.simnorth.SimNorth`.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule

from .transforms import SimTrainTransforms, SimTrainTransformsV2, SimEvalTransforms


class USDataset(Dataset):
    def __init__(self, df, mount_point="./", transform=None, img_column="img_path", repeat_channel=True):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.repeat_channel = repeat_channel

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

        try:
            img = sitk.ReadImage(img_path)
            img_t = torch.tensor(sitk.GetArrayFromImage(img), dtype=torch.float32)

            # Promote single-channel frames to 3 channels so color jitter (hue)
            # is well defined.
            if img.GetNumberOfComponentsPerPixel() == 1 and self.repeat_channel:
                img_t = img_t.unsqueeze(-1).repeat(1, 1, 3)

            img_t = img_t.permute(2, 0, 1)
        except Exception:
            print("Error reading frame: " + img_path, file=sys.stderr)
            img_t = torch.zeros([3, 256, 256], dtype=torch.float32)

        if self.transform:
            img_t = self.transform(img_t)

        return img_t


class USDataModule(LightningDataModule):
    """Reads its dataframes and builds its transforms from the (flat) kwargs,
    mirroring the FAMLI data modules. See :meth:`add_data_specific_args`.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.df_train = self._read_table(os.path.join(self.hparams.mount_point, self.hparams.csv_train))
        self.df_val = self._read_table(os.path.join(self.hparams.mount_point, self.hparams.csv_valid))
        self.df_test = (
            self._read_table(os.path.join(self.hparams.mount_point, self.hparams.csv_test))
            if getattr(self.hparams, "csv_test", None)
            else None
        )

        query = getattr(self.hparams, "query", None)
        if query:
            self.df_train = self.df_train.query(query).reset_index(drop=True)
            self.df_val = self.df_val.query(query).reset_index(drop=True)
            if self.df_test is not None:
                self.df_test = self.df_test.query(query).reset_index(drop=True)

        if self.hparams.train_transform == 2:
            self.train_transform = SimTrainTransformsV2(self.hparams.img_size)
        else:
            self.train_transform = SimTrainTransforms(self.hparams.img_size)
        self.valid_transform = SimEvalTransforms(self.hparams.img_size)
        self.test_transform = self.valid_transform

    @staticmethod
    def _read_table(path):
        if os.path.splitext(path)[1] == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)

    @staticmethod
    def add_data_specific_args(parent_parser):
        group = parent_parser.add_argument_group("USDataModule")
        group.add_argument("--mount_point", default="./", type=str, help="Dataset mount directory")
        group.add_argument("--csv_train", required=True, type=str, help="Train CSV/parquet")
        group.add_argument("--csv_valid", required=True, type=str, help="Validation CSV/parquet")
        group.add_argument("--csv_test", default=None, type=str, help="Test CSV/parquet (optional)")
        group.add_argument("--img_column", default="img_path", type=str, help="Image path column")
        group.add_argument("--img_size", default=224, type=int, help="Square crop size")
        group.add_argument("--query", default=None, type=str, help="Optional pandas query filter")
        group.add_argument("--batch_size", default=256, type=int, help="Batch size")
        group.add_argument("--num_workers", default=4, type=int, help="Dataloader workers")
        group.add_argument("--train_transform", default=0, type=int, help="0=default, 2=V2 transforms")
        group.add_argument("--drop_last", default=1, type=int, help="Drop last incomplete batch")
        group.add_argument("--repeat_channel", default=1, type=int, help="Repeat grayscale frames to 3 channels")
        group.add_argument("--prefetch_factor", default=2, type=int, help="Dataloader prefetch factor")
        return parent_parser

    def setup(self, stage=None):
        repeat_channel = bool(self.hparams.repeat_channel)
        self.train_ds = USDataset(
            self.df_train, self.hparams.mount_point, transform=self.train_transform,
            img_column=self.hparams.img_column, repeat_channel=repeat_channel,
        )
        self.val_ds = USDataset(
            self.df_val, self.hparams.mount_point, transform=self.valid_transform,
            img_column=self.hparams.img_column, repeat_channel=repeat_channel,
        )
        if self.df_test is not None:
            self.test_ds = USDataset(
                self.df_test, self.hparams.mount_point, transform=self.test_transform,
                img_column=self.hparams.img_column, repeat_channel=repeat_channel,
            )

    def _loader(self, ds, shuffle=False):
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
            drop_last=bool(self.hparams.drop_last),
            shuffle=shuffle,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds)

    def test_dataloader(self):
        return self._loader(self.test_ds)

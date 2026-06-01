"""Dataset and LightningDataModule for SimNorth contrastive training.

The dataset reads ultrasound frames listed in a dataframe and applies a *pair*
transform (see :mod:`simnorth.data.transforms`). When a pair transform is used,
each sample is a ``(q, k)`` tuple and the default collate produces a batch
``(x_0, x_1)`` consumed by :class:`simnorth.nets.simnorth.SimNorth`.
"""

import os
import sys

import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule


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
    def __init__(
        self,
        df_train,
        df_val,
        df_test=None,
        mount_point="./",
        batch_size=256,
        num_workers=4,
        img_column="img_path",
        train_transform=None,
        valid_transform=None,
        test_transform=None,
        drop_last=True,
        repeat_channel=True,
        prefetch_factor=2,
    ):
        super().__init__()
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform if test_transform is not None else valid_transform
        self.drop_last = drop_last
        self.repeat_channel = repeat_channel
        self.prefetch_factor = prefetch_factor

    def setup(self, stage=None):
        self.train_ds = USDataset(
            self.df_train, self.mount_point, transform=self.train_transform,
            img_column=self.img_column, repeat_channel=self.repeat_channel,
        )
        self.val_ds = USDataset(
            self.df_val, self.mount_point, transform=self.valid_transform,
            img_column=self.img_column, repeat_channel=self.repeat_channel,
        )
        if self.df_test is not None:
            self.test_ds = USDataset(
                self.df_test, self.mount_point, transform=self.test_transform,
                img_column=self.img_column, repeat_channel=self.repeat_channel,
            )

    def _loader(self, ds, shuffle=False):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            drop_last=self.drop_last,
            shuffle=shuffle,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds)

    def test_dataloader(self):
        return self._loader(self.test_ds)

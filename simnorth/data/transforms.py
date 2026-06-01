"""Augmentation transforms for SimNorth contrastive training.

Each train/eval transform is a *pair* transform: calling it on a single image
returns two independently augmented views ``(q, k)``. With the default
collation this yields a batch ``(x_0, x_1)`` as consumed by
:meth:`SimNorth.training_step`.

Inputs are expected to be 3-channel CHW float tensors in the ``[0, 255]`` range
(see :class:`simnorth.data.dataset.USDataset`).
"""

from torchvision.transforms import v2
from monai.transforms import ScaleIntensityRange


class SimTrainTransforms:
    """Strong augmentation: color jitter, flips, and a random rotate-crop OR
    random resized crop branch."""

    def __init__(self, height: int = 224):
        self.train_transform = v2.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-0.2, 0.2]),
                v2.RandomHorizontalFlip(),
                v2.RandomChoice(
                    [
                        v2.Compose([v2.RandomRotation(180), v2.Pad(64), v2.RandomCrop(height)]),
                        v2.RandomResizedCrop(size=height, scale=(0.4, 1.0), ratio=(0.75, 1.3333333333333333)),
                    ]
                ),
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp), self.train_transform(inp)


class SimTrainTransformsV2:
    """Strong augmentation without random resized crop; always rotate-pad-crop."""

    def __init__(self, height: int = 224):
        self.train_transform = v2.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-0.2, 0.2]),
                v2.RandomHorizontalFlip(),
                v2.Compose([v2.RandomRotation(180), v2.Pad(32), v2.RandomCrop(height)]),
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp), self.train_transform(inp)


class SimEvalTransforms:
    """Deterministic transform: intensity scaling + center crop, returns a pair."""

    def __init__(self, height: int = 224):
        self.eval_transform = v2.Compose(
            [
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.CenterCrop(height),
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp), self.eval_transform(inp)


class SimTestTransforms:
    """Single-view test transform (no pair)."""

    def __init__(self, height: int = 224):
        self.test_transform = SimEvalTransforms(height).eval_transform

    def __call__(self, inp):
        return self.test_transform(inp)

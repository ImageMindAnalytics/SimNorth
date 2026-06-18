"""Augmentation transforms for SimNorth contrastive training.

Each train/eval transform is a *pair* transform: calling it on a single image
returns two independently augmented views ``(q, k)``. With the default
collation this yields a batch ``(x_0, x_1)`` as consumed by
:meth:`SimNorth.training_step`.

Inputs are expected to be 3-channel CHW float tensors in the ``[0, 255]`` range
(see :class:`simnorth.data.dataset.USDataset`).
"""

import cv2
import torch

from torchvision.transforms import v2
from monai.transforms import ScaleIntensityRange


class CLAHE:
    """Contrast Limited Adaptive Histogram Equalization for ultrasound frames.

    Local (tile-wise) contrast equalization. It removes global gain/brightness as
    a discriminative cue and sharpens local anatomical structure, so the encoder
    is pushed toward anatomy rather than acquisition appearance. This is
    deterministic *preprocessing*, not augmentation: apply it identically to both
    contrastive views, to validation, and at clustering time
    (``eval_clusters.py --clahe``) -- otherwise train and eval see different image
    statistics.

    Operates on float tensors in ``[0, 255]`` of shape ``(C, H, W)`` or
    ``(T, C, H, W)``. The first channel (grayscale US) is equalized and the
    result repeated across the ``C`` channels.
    """

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = (int(tile_grid_size), int(tile_grid_size))

    def __call__(self, x):
        single = x.ndim == 3
        if single:
            x = x.unsqueeze(0)  # (1, C, H, W)
        # cv2's CLAHE object isn't thread/fork-shareable; build one per call (cheap).
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        out = torch.empty_like(x)
        for t in range(x.shape[0]):
            g = x[t, 0].clamp(0, 255).round().to(torch.uint8).cpu().numpy()
            eq = torch.from_numpy(clahe.apply(g)).to(x.dtype)  # (H, W) in [0, 255]
            out[t] = eq.unsqueeze(0).expand(x.shape[1], -1, -1)
        return out[0] if single else out


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


class SimTrainTransformsV3:
    """CLAHE-enhanced augmentation for ultrasound.

    Local-contrast equalization (CLAHE) suppresses the gain/brightness/overlay
    shortcut up front; the usual rotate-pad-crop + color jitter follow, and a
    light random Gaussian blur discourages speckle/pixel-level shortcuts so
    alignment learns anatomical structure. CLAHE is deterministic, but jitter,
    crop, flip, and blur remain independent per view.
    """

    def __init__(self, height: int = 224, clahe_clip: float = 2.0, clahe_grid: int = 8):
        self.train_transform = v2.Compose(
            [
                CLAHE(clip_limit=clahe_clip, tile_grid_size=clahe_grid),
                ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
                v2.ColorJitter(brightness=[0.5, 1.5], contrast=[0.5, 1.5], saturation=[0.5, 1.5], hue=[-0.2, 0.2]),
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(30),
                # No pad: frames are larger than the crop, so RandomCrop draws a
                # window of real content (translation jitter) with no synthetic
                # black border -- eval's CenterCrop(height) is the center of this
                # same distribution. pad_if_needed guards rare smaller frames.
                v2.RandomCrop(height, pad_if_needed=True),
                v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp), self.train_transform(inp)


class SimEvalTransforms:
    """Deterministic transform: (optional CLAHE) + intensity scaling + center
    crop, returns a pair. ``clahe`` must match the training transform."""

    def __init__(self, height: int = 224, clahe: bool = False, clahe_clip: float = 2.0, clahe_grid: int = 8):
        steps = []
        if clahe:
            steps.append(CLAHE(clip_limit=clahe_clip, tile_grid_size=clahe_grid))
        steps += [
            ScaleIntensityRange(a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0),
            v2.CenterCrop(height),
        ]
        self.eval_transform = v2.Compose(steps)

    def __call__(self, inp):
        return self.eval_transform(inp), self.eval_transform(inp)


class SimTestTransforms:
    """Single-view test transform (no pair). ``clahe`` must match training."""

    def __init__(self, height: int = 224, clahe: bool = False, clahe_clip: float = 2.0, clahe_grid: int = 8):
        self.test_transform = SimEvalTransforms(
            height, clahe=clahe, clahe_clip=clahe_clip, clahe_grid=clahe_grid
        ).eval_transform

    def __call__(self, inp):
        return self.test_transform(inp)

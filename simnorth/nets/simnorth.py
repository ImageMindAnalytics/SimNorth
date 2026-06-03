"""SimNorth contrastive self-supervised model.

SimNorth learns ultrasound frame embeddings on a hypersphere. Two augmented
views of the same frame are pulled together (alignment), randomly paired views
are pushed apart with a rank-weighted penalty (uniformity), and a set of fixed
"light house" anchors on the sphere attract their nearest embeddings so the
learned manifold organizes around them.
"""

import pickle

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

import lightning.pytorch as pl


class ProjectionHead(nn.Module):
    """MLP projection head that maps encoder features onto the unit hypersphere.

    The output is passed through ``abs`` before normalization so embeddings live
    on the positive orthant of the sphere (where the light houses are sampled).
    """

    def __init__(self, input_dim=1280, hidden_dim=1280, output_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.abs(x)
        return F.normalize(x, dim=1)


class GaussianNoise(nn.Module):
    """Additive Gaussian noise, active only in training mode."""

    def __init__(self, mean=0.0, std=0.05):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        if self.training:
            return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)
        return x


class SimNorth(pl.LightningModule):
    """Self-supervised contrastive model with hypersphere "light house" anchors.

    All arguments are passed as keyword arguments (typically ``**vars(args)``
    from the trainer) and captured via ``save_hyperparameters``. See
    :meth:`add_model_specific_args` for the available hyperparameters:

        base_encoder: Any ``torchvision.models`` constructor name (e.g.
            ``efficientnet_b0``, ``resnet18``). The classifier/fc head is
            replaced with a :class:`ProjectionHead`.
        emb_dim: Dimensionality of the embedding on the hypersphere.
        hidden_dim: Hidden width of the projection head.
        n_lights: Number of light house anchors (ignored if ``lights`` points to
            a pickled set of anchors).
        lights: Optional path to a pickle holding ``(n_lights, emb_dim)`` anchor
            points. If ``None``, anchors are drawn uniformly in ``[0, 1)``.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        epochs: Used by the cosine annealing scheduler (``T_max``).
        w: Scale of the rank-based weighting on the contrastive (push) term.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        template_model = getattr(torchvision.models, self.hparams.base_encoder)
        self.convnet = template_model(num_classes=4 * self.hparams.emb_dim)

        proj_head = ProjectionHead(
            input_dim=4 * self.hparams.emb_dim,
            hidden_dim=self.hparams.hidden_dim,
            output_dim=self.hparams.emb_dim,
        )
        if hasattr(self.convnet, "classifier"):
            self.convnet.classifier = nn.Sequential(self.convnet.classifier, proj_head)
        elif hasattr(self.convnet, "fc"):
            self.convnet.fc = nn.Sequential(self.convnet.fc, proj_head)
        else:
            raise ValueError(
                f"Unsupported base_encoder '{self.hparams.base_encoder}': no classifier/fc head found."
            )

        self.loss = nn.CosineSimilarity()

        self.noise_transform = nn.Sequential(GaussianNoise())

        lights = getattr(self.hparams, "lights", None)
        if lights is None:
            light_house = torch.rand(self.hparams.n_lights, self.hparams.emb_dim)
        elif isinstance(lights, str):
            with open(lights, "rb") as f:
                light_house = torch.as_tensor(pickle.load(f), dtype=torch.float32)
        else:
            light_house = torch.as_tensor(lights, dtype=torch.float32)

        self.register_buffer("light_house", light_house)

        # Scale the light-house jitter to half the smallest inter-anchor distance
        # so the noise never lets one anchor wander into another's basin.
        min_l = torch.tensor(float("inf"))
        for idx, l in enumerate(light_house):
            lights_ex = torch.cat([light_house[:idx], light_house[idx + 1:]])
            min_l = torch.minimum(min_l, torch.min(torch.sum(torch.square(l - lights_ex), dim=1)))

        self.noise_transform_lights = nn.Sequential(GaussianNoise(mean=0.0, std=min_l.item() / 2.0))

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SimNorth")
        group.add_argument("--base_encoder", default="efficientnet_b0", type=str, help="torchvision encoder")
        group.add_argument("--emb_dim", default=128, type=int, help="Embedding dimension")
        group.add_argument("--hidden_dim", default=64, type=int, help="Projection head hidden dim")
        group.add_argument("--n_lights", default=64, type=int, help="Number of light house anchors")
        group.add_argument("--lights", default=None, type=str, help="Pickle file with light house anchors")
        group.add_argument("--w", default=4.0, type=float, help="Weight scale for the contrastive (push) term")
        group.add_argument("--lr", "--learning-rate", default=1e-4, type=float, help="Learning rate")
        group.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")
        return parent_parser

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        max_epochs = getattr(self.hparams, "epochs", None) or getattr(self.hparams, "max_epochs", None) or 200
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def compute_loss(self, x_0, x_1, mode):
        batch_size = x_0.size(0)

        x = torch.cat([x_0, x_1], dim=0)
        z = self(self.noise_transform(x))
        z_0, z_1 = torch.split(z, batch_size)

        # --- Alignment: the two views of the same frame should match ---
        loss_proj = self.loss(z_0, z_1)
        loss_proj_mean = torch.mean(loss_proj)
        loss_proj_std = torch.std(loss_proj)
        loss_proj = torch.sum(torch.square(1.0 - loss_proj))

        # --- Uniformity: random pairs are pushed apart, rank-weighted so the
        #     most-similar (likely true-negative) random pairs are penalized most.
        r = torch.randperm(batch_size)
        z_0_r = z_0[r]
        loss_proj_c = self.loss(z_0_r, z_1)
        loss_proj_c_mean = torch.mean(loss_proj_c)
        loss_proj_c_std = torch.std(loss_proj_c)

        loss_proj_c_sorted_i = torch.argsort(loss_proj_c)  # ascending: most different first
        loss_proj_c = loss_proj_c[loss_proj_c_sorted_i]
        w = torch.square(torch.arange(batch_size, device=self.device) / batch_size - 1.0) * self.hparams.w
        loss_proj_c = torch.sum(w * torch.square(loss_proj_c))

        # --- North / light house: each anchor attracts its single closest
        #     embedding from each view, organizing the manifold around anchors.
        loss_north = []
        light_house = self.noise_transform_lights(self.light_house)
        for lh in light_house:
            l_north = self.loss(z_0, lh)
            nearest_z0 = torch.argsort(l_north)[-1]  # most similar embedding
            loss_north.append(1.0 - l_north[nearest_z0])  # pull it closer

            l_north = self.loss(z_1, lh)
            nearest_z1 = torch.argsort(l_north)[-1]
            loss_north.append(1.0 - l_north[nearest_z1])  # pull it closer

        loss_north = torch.stack(loss_north)
        loss_north_mean = torch.mean(loss_north)
        loss_north_std = torch.std(loss_north)
        loss_north = torch.square(torch.sum(loss_north))

        loss = loss_proj + loss_proj_c + loss_north

        self.log(mode + "_loss_proj", loss_proj, sync_dist=True)
        self.log(mode + "_loss_proj_mean", loss_proj_mean, sync_dist=True)
        self.log(mode + "_loss_proj_std", loss_proj_std, sync_dist=True)
        self.log(mode + "_loss_proj_c", loss_proj_c, sync_dist=True)
        self.log(mode + "_loss_proj_c_mean", loss_proj_c_mean, sync_dist=True)
        self.log(mode + "_loss_proj_c_std", loss_proj_c_std, sync_dist=True)
        self.log(mode + "_loss_north_mean", loss_north_mean, sync_dist=True)
        self.log(mode + "_loss_north_std", loss_north_std, sync_dist=True)
        self.log(mode + "_loss_north", loss_north, sync_dist=True)
        self.log(mode + "_loss", loss, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        x_0, x_1 = batch
        return self.compute_loss(x_0, x_1, mode="train")

    def validation_step(self, batch, batch_idx):
        x_0, x_1 = batch
        self.compute_loss(x_0, x_1, mode="val")

    def forward(self, x):
        return self.convnet(x)

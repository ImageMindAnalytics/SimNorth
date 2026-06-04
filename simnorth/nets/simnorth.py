"""SimNorth contrastive self-supervised model.

SimNorth learns ultrasound frame embeddings on a hypersphere. Two augmented
views of the same frame are pulled together (alignment), randomly paired views
are pushed apart with a rank-weighted penalty (uniformity), and a set of fixed
"light house" anchors on the sphere attract their nearest embeddings so the
learned manifold organizes around them.

Validation also reports a cluster-count metric: after ``warmup_epochs``,
validation embeddings are tracked with a ``CatMetric`` (gathered across all
ranks for multi-GPU training); on rank 0 a random subset of up to
``n_cluster_samples`` is taken and the silhouette-optimal number of KMeans
clusters is logged as ``val_n_clusters`` (monitor with ``mode="max"`` to favor
more well-separated clusters).
"""

import pickle

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

import lightning.pytorch as pl
from torchmetrics.aggregation import CatMetric


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
            a pickled set of anchors). Set ``<= 0`` (with no ``lights`` pickle) to
            disable the light house term entirely.
        lights: Optional path to a pickle holding ``(n_lights, emb_dim)`` anchor
            points. If ``None`` and ``n_lights > 0``, anchors are drawn uniformly
            in ``[0, 1)``; if ``None`` and ``n_lights <= 0`` the term is disabled.
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

        # The light house (north) term is optional. It is enabled when anchors
        # are supplied via the ``lights`` pickle, or when ``n_lights > 0`` (random
        # anchors). With no pickle and ``n_lights <= 0`` the model trains on the
        # alignment + uniformity objectives only.
        lights = getattr(self.hparams, "lights", None)
        if lights is None and self.hparams.n_lights <= 0:
            light_house = None
        elif lights is None:
            light_house = torch.rand(self.hparams.n_lights, self.hparams.emb_dim)
        elif isinstance(lights, str):
            with open(lights, "rb") as f:
                light_house = torch.as_tensor(pickle.load(f), dtype=torch.float32)
        else:
            light_house = torch.as_tensor(lights, dtype=torch.float32)

        self.register_buffer("light_house", light_house)

        if light_house is None:
            self.noise_transform_lights = None
        else:
            # Scale the light-house jitter to half the smallest inter-anchor
            # distance so the noise never lets one anchor wander into another's
            # basin.
            min_l = torch.tensor(float("inf"))
            for idx, l in enumerate(light_house):
                lights_ex = torch.cat([light_house[:idx], light_house[idx + 1:]])
                min_l = torch.minimum(min_l, torch.min(torch.sum(torch.square(l - lights_ex), dim=1)))

            self.noise_transform_lights = nn.Sequential(GaussianNoise(mean=0.0, std=min_l.item() / 2.0))

        # Validation embeddings for the cluster-count metric. CatMetric gathers
        # and concatenates across all ranks (DDP-safe) when computed.
        self.val_features = CatMetric()

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SimNorth")
        group.add_argument("--base_encoder", default="efficientnet_b0", type=str, help="torchvision encoder")
        group.add_argument("--emb_dim", default=128, type=int, help="Embedding dimension")
        group.add_argument("--hidden_dim", default=64, type=int, help="Projection head hidden dim")
        group.add_argument("--n_lights", default=64, type=int, help="Number of light house anchors (<=0 disables the light house term)")
        group.add_argument("--lights", default=None, type=str, help="Pickle file with light house anchors")
        group.add_argument("--w", default=4.0, type=float, help="Weight scale for the contrastive (push) term")
        group.add_argument("--lr", "--learning-rate", default=1e-4, type=float, help="Learning rate")
        group.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")

        # Cluster-count validation: after a warmup, reservoir-sample validation
        # embeddings and report the silhouette-optimal number of clusters.
        group.add_argument("--warmup_epochs", default=10, type=int, help="Start cluster-count validation after this many epochs")
        group.add_argument("--n_cluster_samples", default=1024, type=int, help="Embeddings to reservoir-sample for cluster validation (<=0 disables)")
        group.add_argument("--n_clusters_min", default=2, type=int, help="Minimum k for the silhouette search")
        group.add_argument("--n_clusters_max", default=20, type=int, help="Maximum k for the silhouette search")
        return parent_parser

    @staticmethod
    def suggest_hyper_params(trial):
        """Suggest a hyperparameter set for an Optuna trial.

        The returned dict overrides the matching CLI arguments for the trial.
        The backbone (``base_encoder``) is part of the search space.
        """
        return {
            "base_encoder": trial.suggest_categorical(
                "base_encoder",
                ["efficientnet_b0", "efficientnet_b1", "efficientnet_v2_s",
                 "resnet18", "resnet34", "resnet50",
                 "convnext_tiny", "convnext_small"],
            ),
            "emb_dim": trial.suggest_categorical("emb_dim", [64, 128, 256]),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "n_lights": trial.suggest_int("n_lights", 0, 128, step=16),
            "w": trial.suggest_float("w", 0.5, 8.0, log=True),
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        }

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

        loss = loss_proj + loss_proj_c

        self.log(mode + "_loss_proj", loss_proj, sync_dist=True)
        self.log(mode + "_loss_proj_mean", loss_proj_mean, sync_dist=True)
        self.log(mode + "_loss_proj_std", loss_proj_std, sync_dist=True)
        self.log(mode + "_loss_proj_c", loss_proj_c, sync_dist=True)
        self.log(mode + "_loss_proj_c_mean", loss_proj_c_mean, sync_dist=True)
        self.log(mode + "_loss_proj_c_std", loss_proj_c_std, sync_dist=True)

        # --- North / light house (optional): each anchor attracts its single
        #     closest embedding from each view, organizing the manifold around
        #     anchors. Skipped when no light house is configured.
        if self.light_house is not None:
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

            loss = loss + loss_north

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

        # Track clean (un-jittered) embeddings for the cluster-count metric.
        # CatMetric accumulates per rank and concatenates across ranks on compute.
        if self._cluster_validation_active():
            self.val_features.update(self(x_0))

    def _cluster_validation_active(self):
        """Cluster validation runs only after the warmup, and when enabled."""
        return self.hparams.n_cluster_samples > 0 and self.current_epoch >= self.hparams.warmup_epochs

    def on_validation_epoch_end(self):
        if self.hparams.n_cluster_samples <= 0:
            return

        # Keep val_n_clusters present from epoch 0 (so EarlyStopping/ModelCheckpoint
        # can monitor it), but neutral until the warmup starts collecting features.
        if not self._cluster_validation_active():
            self.log("val_n_clusters", 0.0, sync_dist=True, prog_bar=True)
            return

        # compute() gathers features from every rank; it is a collective and must
        # run on all ranks. The silhouette search itself runs only on rank 0.
        features = self.val_features.compute()
        self.val_features.reset()

        n_clusters, silhouette = 0.0, -1.0
        if self.trainer.is_global_zero:
            n_clusters, silhouette = self._optimal_n_clusters(features)

        # Broadcast the rank-0 result so the monitored metric is identical on
        # every rank (EarlyStopping/ModelCheckpoint run on all ranks).
        n_clusters = self.trainer.strategy.broadcast(n_clusters, src=0)
        silhouette = self.trainer.strategy.broadcast(silhouette, src=0)
        self.log("val_n_clusters", n_clusters, sync_dist=False, prog_bar=True)
        self.log("val_silhouette", silhouette, sync_dist=False)

    def _optimal_n_clusters(self, features):
        """Silhouette-optimal KMeans cluster count over a random subset of the
        gathered validation embeddings. Returns ``(n_clusters, silhouette)``."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        feats = features.detach().float().cpu()
        n = self.hparams.n_cluster_samples
        if 0 < n < feats.shape[0]:
            feats = feats[torch.randperm(feats.shape[0])[:n]]
        feats = feats.numpy()

        if feats.shape[0] <= self.hparams.n_clusters_min:
            return 0.0, -1.0

        k_max = min(self.hparams.n_clusters_max, feats.shape[0] - 1)
        best_k, best_score = 0, -1.0
        for k in range(self.hparams.n_clusters_min, k_max + 1):
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(feats)
            # KMeans can leave clusters empty on degenerate (e.g. collapsed,
            # untrained) embeddings; silhouette needs 2..n_samples-1 labels.
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(feats, labels)
            if score > best_score:
                best_score, best_k = score, k
        return float(best_k), float(best_score)

    def forward(self, x):
        return self.convnet(x)

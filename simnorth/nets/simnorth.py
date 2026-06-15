"""SimNorth contrastive self-supervised model.

SimNorth learns ultrasound frame embeddings on the unit hypersphere. Two
augmented views of the same frame are pulled together (alignment), randomly
paired views are pushed apart with a rank-weighted penalty (uniformity), and a
prototypical-contrastive term (ProtoNCE) pulls each embedding toward the
data-driven cluster centroid it belongs to.

The prototypes follow the Expectation-Maximization scheme of Prototypical
Contrastive Learning (Li et al., ICLR 2021, ``simnorth/docs/2005.04966v5.pdf``):
at the end of each validation epoch (E-step), embeddings gathered across all
ranks with a ``CatMetric`` are clustered with KMeans into ``M`` granularities,
and each centroid's concentration ``phi`` is estimated; during the next epoch
(M-step) the ProtoNCE term contrastively pulls embeddings toward their assigned
centroid scaled by ``phi``. Clustering is bootstrapped by ``warmup_epochs`` of
alignment + uniformity only, so the prototype term is inactive (contributes 0)
until then.

The same E-step also reports a silhouette-optimal cluster-count metric as
``val_n_clusters`` (monitor with ``mode="max"`` to favor well-separated
clusters) and ``val_silhouette``.
"""

import math

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision

import lightning.pytorch as pl
from torchmetrics.aggregation import CatMetric


class ProjectionHead(nn.Module):
    """MLP projection head that maps encoder features onto the unit hypersphere.

    The output is L2-normalized so embeddings live on the full unit sphere
    (data-driven prototypes are not confined to any orthant).
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
    """Self-supervised contrastive model with data-driven prototypes (ProtoNCE).

    All arguments are passed as keyword arguments (typically ``**vars(args)``
    from the trainer) and captured via ``save_hyperparameters``. See
    :meth:`add_model_specific_args` for the available hyperparameters:

        base_encoder: Any ``torchvision.models`` constructor name (e.g.
            ``efficientnet_b0``, ``resnet18``). The classifier/fc head is
            replaced with a :class:`ProjectionHead`.
        emb_dim: Dimensionality of the embedding on the hypersphere.
        hidden_dim: Hidden width of the projection head.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        epochs: Used by the cosine annealing scheduler (``T_max``).
        w: Scale of the rank-based weighting on the contrastive (push) term.
        proto_clusters: Comma-separated list of cluster counts ``K`` for the
            ProtoNCE term (e.g. ``"8,16,32"``). Empty derives ``{k*, 2k*, 4k*}``
            from the silhouette-optimal ``k*``.
        proto_samples: Embeddings sampled for the prototype k-means E-step
            (``<= 0`` uses all gathered embeddings). Decoupled from the O(N^2)
            silhouette metric's ``n_cluster_samples`` so prototypes can be
            estimated from many more points without the quadratic cost.
        proto_tau: Instance temperature / mean to which each clustering's
            concentrations ``phi`` are normalized.
        proto_alpha: Concentration smoothing (eq. 12) so small clusters do not
            get an overly large ``phi``.
        proto_weight: Weight of the ProtoNCE term in the total loss.
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

        # Prototypes (cluster centroids) and their concentrations, re-estimated
        # each validation epoch (E-step). One (centroids, phi) pair per
        # granularity in ``K``. Empty until the warmup completes; the ProtoNCE
        # term contributes 0 while empty. Not checkpointed -- recomputed on the
        # first validation epoch after a resume.
        self._prototypes = []      # list of (k_m, emb_dim) tensors, L2-normalized
        self._concentrations = []  # list of (k_m,) tensors

        # Validation embeddings for the E-step and cluster-count metric.
        # CatMetric gathers and concatenates across all ranks (DDP-safe) on compute.
        self.val_features = CatMetric()

    @staticmethod
    def add_model_specific_args(parent_parser):
        group = parent_parser.add_argument_group("SimNorth")
        group.add_argument("--base_encoder", default="efficientnet_b0", type=str, help="torchvision encoder")
        group.add_argument("--emb_dim", default=128, type=int, help="Embedding dimension")
        group.add_argument("--hidden_dim", default=64, type=int, help="Projection head hidden dim")
        group.add_argument("--w", default=4.0, type=float, help="Weight scale for the contrastive (push) term")
        group.add_argument("--lr", "--learning-rate", default=1e-4, type=float, help="Learning rate")
        group.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")

        # ProtoNCE (prototypical contrastive) term. Clustering is gated by
        # ``warmup_epochs``; the term is inactive until then.
        group.add_argument("--proto_clusters", default="", type=str, help="Comma-separated cluster counts K (e.g. '8,16,32'); empty derives {k*,2k*,4k*} from the silhouette-optimal k*")
        group.add_argument("--proto_samples", default=4096, type=int, help="Embeddings to sample for the prototype k-means E-step (<=0 uses all gathered embeddings). Decoupled from the O(N^2) silhouette metric's --n_cluster_samples.")
        group.add_argument("--proto_tau", default=0.1, type=float, help="Instance temperature / mean concentration the phi's are normalized to")
        group.add_argument("--proto_alpha", default=10.0, type=float, help="Concentration smoothing (eq. 12)")
        group.add_argument("--proto_weight", default=1.0, type=float, help="Weight of the ProtoNCE term in the total loss")

        # Cluster-count validation: after a warmup, sample validation embeddings
        # and report the silhouette-optimal number of clusters. This is also the
        # master switch / warmup gate for the prototype E-step (see _proto_loss).
        group.add_argument("--warmup_epochs", default=10, type=int, help="Start clustering (silhouette metric + prototype E-step) after this many epochs")
        group.add_argument("--n_cluster_samples", default=4096, type=int, help="Embeddings to sample for the O(N^2) silhouette metric (<=0 disables clustering, including the prototype term)")
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
            "w": trial.suggest_float("w", 0.5, 8.0, log=True),
            "proto_weight": trial.suggest_float("proto_weight", 0.1, 4.0, log=True),
            "proto_tau": trial.suggest_float("proto_tau", 0.05, 0.5, log=True),
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

        # Only sync logged metrics across ranks for the epoch-level validation
        # aggregates; per-step training logs avoid the all-reduce overhead.
        sync = mode != "train"

        x = torch.cat([x_0, x_1], dim=0)
        z = self(self.noise_transform(x))
        z_0, z_1 = torch.split(z, batch_size)

        # --- Alignment: the two views of the same frame should match ---
        loss_proj = self.loss(z_0, z_1)
        loss_proj_mean = torch.mean(loss_proj)
        loss_proj_std = torch.std(loss_proj)
        loss_proj = torch.sum(torch.square(1.0 - loss_proj))

        # --- Uniformity: random pairs are pushed apart, rank-weighted so the
        #     most-similar random pairs (likely false negatives -- semantically
        #     related, so we don't want to force them apart) are penalized least,
        #     while the most-different pairs are penalized most.
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

        self.log(mode + "_loss_proj", loss_proj, sync_dist=sync)
        self.log(mode + "_loss_proj_mean", loss_proj_mean, sync_dist=sync)
        self.log(mode + "_loss_proj_std", loss_proj_std, sync_dist=sync)
        self.log(mode + "_loss_proj_c", loss_proj_c, sync_dist=sync)
        self.log(mode + "_loss_proj_c_mean", loss_proj_c_mean, sync_dist=sync)
        self.log(mode + "_loss_proj_c_std", loss_proj_c_std, sync_dist=sync)

        # --- ProtoNCE (north): pull every embedding toward its assigned cluster
        #     centroid, scaled by the centroid's concentration. Inactive (no
        #     prototypes) until the warmup completes and the first E-step runs.
        if self._prototypes:
            loss_proto = self._proto_loss(z)
            loss = loss + self.hparams.proto_weight * loss_proto
            self.log(mode + "_loss_proto", loss_proto, sync_dist=sync)

        self.log(mode + "_loss", loss, sync_dist=sync)

        return loss

    def _proto_loss(self, z):
        """ProtoNCE prototype term (eq. 11, prototype part) averaged over the
        ``M`` clustering granularities.

        Each embedding is hard-assigned to its nearest centroid (cosine) and a
        cross-entropy pulls it toward that centroid while pushing it from the
        others, with per-centroid concentration ``phi`` acting as temperature.
        Centroids carry no gradient (fixed during the M-step)."""
        total = z.new_zeros(())
        for centroids, phi in zip(self._prototypes, self._concentrations):
            centroids = centroids.to(z.device)
            phi = phi.to(z.device)
            logits = (z @ centroids.t()) / phi.clamp_min(1e-4)  # (N, k)
            assign = logits.argmax(dim=1)
            total = total + F.cross_entropy(logits, assign)
        return total / len(self._prototypes)

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
            self.log("val_n_clusters", 0.0, sync_dist=False, prog_bar=True)
            self.log("val_silhouette", -1.0, sync_dist=False)
            return

        # compute() gathers features from every rank; it is a collective and must
        # run on all ranks. The silhouette search itself runs only on rank 0.
        features = self.val_features.compute()
        self.val_features.reset()

        n_clusters, silhouette = 0.0, -1.0
        prototypes, concentrations = [], []
        if self.trainer.is_global_zero:
            n_clusters, silhouette = self._optimal_n_clusters(features)
            # E-step: cluster the gathered embeddings into the ProtoNCE
            # granularities and estimate each centroid's concentration.
            prototypes, concentrations = self._compute_prototypes(features, n_clusters)

        # Broadcast the rank-0 results so the monitored metric and the prototypes
        # are identical on every rank (EarlyStopping/ModelCheckpoint and the
        # M-step run on all ranks). Prototypes travel as CPU tensors.
        n_clusters = self.trainer.strategy.broadcast(n_clusters, src=0)
        silhouette = self.trainer.strategy.broadcast(silhouette, src=0)
        prototypes = self.trainer.strategy.broadcast(prototypes, src=0)
        concentrations = self.trainer.strategy.broadcast(concentrations, src=0)
        self._prototypes = [p.to(self.device) for p in prototypes]
        self._concentrations = [c.to(self.device) for c in concentrations]

        self.log("val_n_clusters", n_clusters, sync_dist=False, prog_bar=True)
        self.log("val_silhouette", silhouette, sync_dist=False)
        self.log("val_n_prototypes", float(sum(p.shape[0] for p in self._prototypes)), sync_dist=False)

    def _compute_prototypes(self, features, k_star):
        """E-step: cluster the gathered embeddings into each granularity in
        ``K`` and estimate per-centroid concentrations. Runs on rank 0 only;
        returns ``(prototypes, concentrations)`` as lists of CPU tensors for
        broadcasting. Centroids are L2-normalized so cosine logits are well
        scaled."""
        from torch_kmeans import KMeans

        feats = features.detach().float().to(self.device)
        # Sample independently of the silhouette metric: the prototype k-means is
        # O(N*k*iters), not O(N^2), so it scales to many more samples cheaply.
        n = self.hparams.proto_samples
        if 0 < n < feats.shape[0]:
            feats = feats[torch.randperm(feats.shape[0], device=feats.device)[:n]]
        feats = F.normalize(feats, dim=1)

        K = self._proto_K(k_star, feats.shape[0])
        if not K:
            return [], []

        x = feats.unsqueeze(0)  # torch-kmeans expects (B, N, D)
        prototypes, concentrations = [], []
        for k in K:
            res = KMeans(n_clusters=k, num_init=10, seed=0, verbose=False)(x)
            labels = res.labels[0]
            centroids = F.normalize(res.centers[0], dim=1)
            phi = self._concentration(feats, labels, centroids)
            prototypes.append(centroids.detach().cpu())
            concentrations.append(phi.detach().cpu())
        return prototypes, concentrations

    def _proto_K(self, k_star, n_samples):
        """Resolve the list of cluster counts ``K`` for ProtoNCE. Uses
        ``--proto_clusters`` when given, otherwise derives ``{k*, 2k*, 4k*}``
        from the silhouette-optimal ``k*``. Each ``k`` is kept only if
        ``2 <= k < n_samples``."""
        if self.hparams.proto_clusters:
            ks = [int(tok) for tok in self.hparams.proto_clusters.split(",") if tok.strip()]
        else:
            k = int(k_star)
            ks = [k, 2 * k, 4 * k] if k >= 2 else []
        return sorted({k for k in ks if 2 <= k < n_samples})

    def _concentration(self, feats, labels, centroids):
        """Per-centroid concentration ``phi`` (eq. 12):
        ``phi = sum_z ||v_z - c||^2 / (Z * log(Z + alpha))`` over the ``Z``
        members of each cluster, then normalized so the set's mean is
        ``proto_tau``. Empty/singleton clusters take the set mean."""
        alpha = self.hparams.proto_alpha
        tau = self.hparams.proto_tau
        k = centroids.shape[0]
        phi = centroids.new_full((k,), float("nan"))
        for c in range(k):
            mask = labels == c
            Z = int(mask.sum())
            if Z <= 1:
                continue
            d2 = torch.sum((feats[mask] - centroids[c]) ** 2)
            phi[c] = d2 / (Z * math.log(Z + alpha))

        valid = ~torch.isnan(phi)
        if not valid.any():
            return centroids.new_full((k,), tau)
        phi = torch.where(torch.isnan(phi), phi[valid].mean(), phi)
        return phi * (tau / phi.mean().clamp_min(1e-12))

    def _optimal_n_clusters(self, features):
        """Silhouette-optimal KMeans cluster count over a random subset of the
        gathered validation embeddings. Returns ``(n_clusters, silhouette)``.

        Clustering uses GPU-friendly ``torch_kmeans.KMeans`` and the silhouette
        coefficient is computed in pure torch (see :meth:`_silhouette_score`),
        so the whole search stays on-device with no sklearn dependency."""
        from torch_kmeans import KMeans

        feats = features.detach().float().to(self.device)
        n = self.hparams.n_cluster_samples
        if 0 < n < feats.shape[0]:
            feats = feats[torch.randperm(feats.shape[0], device=feats.device)[:n]]

        if feats.shape[0] <= self.hparams.n_clusters_min:
            return 0.0, -1.0

        k_max = min(self.hparams.n_clusters_max, feats.shape[0] - 1)
        # torch-kmeans expects a batch of datasets, shape (B, N, D).
        x = feats.unsqueeze(0)
        best_k, best_score = 0, -1.0
        for k in range(self.hparams.n_clusters_min, k_max + 1):
            # num_init/seed mirror sklearn's n_init/random_state; the default
            # LpDistance(p_norm=2) matches sklearn's euclidean metric.
            model = KMeans(n_clusters=k, num_init=10, seed=0, verbose=False)
            labels = model(x).labels[0]
            # KMeans can leave clusters empty on degenerate (e.g. collapsed,
            # untrained) embeddings; silhouette needs at least 2 populated clusters.
            if torch.unique(labels).numel() < 2:
                continue
            score = self._silhouette_score(feats, labels)
            if torch.isnan(score):
                continue
            score = float(score)
            if score > best_score:
                best_score, best_k = score, k
        return float(best_k), float(best_score)

    @staticmethod
    def _silhouette_score(feats, labels):
        """Mean silhouette coefficient (euclidean), computed entirely in torch.

        Mirrors ``sklearn.metrics.silhouette_score``: each sample scores
        ``s = (b - a) / max(a, b)`` where ``a`` is its mean intra-cluster
        distance and ``b`` its mean distance to the nearest other cluster;
        singleton clusters score 0 and the per-sample scores are averaged.
        Returns ``nan`` when fewer than two clusters are present.
        """
        unique = torch.unique(labels)  # sorted, so searchsorted yields 0..K-1
        n_clusters = unique.numel()
        if n_clusters < 2:
            return feats.new_tensor(float("nan"))

        label_idx = torch.searchsorted(unique, labels)
        onehot = F.one_hot(label_idx, n_clusters).to(feats.dtype)  # (N, K)
        cluster_sizes = onehot.sum(dim=0)  # (K,)

        dist = torch.cdist(feats, feats)  # (N, N) euclidean
        dist_to_cluster = dist @ onehot  # (N, K) summed distance to each cluster

        own = onehot.bool()  # (N, K), one True per row
        own_size = cluster_sizes[label_idx]  # (N,)

        # a: mean intra-cluster distance (self-distance is 0, so excluded).
        a = dist_to_cluster[own] / (own_size - 1).clamp(min=1)

        # b: smallest mean distance to any *other* cluster.
        mean_to_cluster = dist_to_cluster / cluster_sizes.clamp(min=1)  # (N, K)
        mean_to_cluster = mean_to_cluster.masked_fill(own, float("inf"))
        b = mean_to_cluster.min(dim=1).values  # (N,)

        denom = torch.maximum(a, b)
        s = torch.where(denom > 0, (b - a) / denom, torch.zeros_like(denom))
        # Singleton clusters contribute 0 by convention.
        s = torch.where(own_size > 1, s, torch.zeros_like(s))
        return s.mean()

    def forward(self, x):
        return self.convnet(x)

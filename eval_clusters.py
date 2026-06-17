"""Cluster-evaluate a trained SimNorth model over blind-sweep cines.

Loads a checkpoint, embeds the blind-sweep cines listed in a CSV/parquet, sweeps
KMeans over a range of ``k`` with the silhouette score, then at the chosen ``k``
writes the outputs below. By default **every frame** of every cine is embedded;
pass ``--n_samples N`` to instead shuffle the sweeps and take ``--frames_per_sweep``
random frames from each until ``N`` total frames are collected.

  * ``silhouette.png``   - silhouette vs. k, with the chosen k marked
  * ``clusters_grid.png``- montage: one row per cluster, the frames nearest its
                           centroid (the cluster's representative images)
  * ``clusters.csv``     - dataframe ``file_path,index,cluster_label`` with one
                           row per frame (``index`` is the frame index in the cine)
  * ``summary.json``     - chosen k, silhouette, per-cluster sizes

Frame reading mirrors :class:`simnorth.data.dataset.USDatasetBlindSweep` (first
component of multi-channel cines, repeat-to-3-channels) but keeps **all** frames
in order instead of sampling, and applies the deterministic single-view
:class:`simnorth.data.transforms.SimTestTransforms`. Clustering/silhouette reuse
the same ``torch_kmeans`` + pure-torch silhouette the model uses in training.

Example:

    python eval_clusters.py \
        --model checkpoints/last.ckpt \
        --csv sweeps.parquet --mount_point /data --img_column file_path \
        --img_size 256 --n_clusters_min 2 --n_clusters_max 40 \
        --out eval_out
"""

import argparse
import json
import os
import sys
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_kmeans import KMeans

from simnorth import SimNorth
from simnorth.data.dataset import _worker_init
from simnorth.data.transforms import SimTestTransforms

torch.multiprocessing.set_sharing_strategy('file_system')

def _read_table(path):
    if os.path.splitext(path)[1] == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _read_cine(path, args, transform, n_frames=None):
    """Read frames of a blind-sweep cine. Returns ``(frames, orig_idx)``
    where ``frames`` is ``(T', 3, H, W)`` after the eval transform and
    ``orig_idx`` are the original frame indices kept (honoring ``frame_stride``).
    If ``n_frames`` is set and the cine has more frames than that, a random
    subset of ``n_frames`` (seeded per-path for reproducibility) is kept.
    Returns ``(None, None)`` on a read error."""
    full = os.path.join(args.mount_point, path)
    try:
        img = sitk.ReadImage(full)
        arr = torch.from_numpy(sitk.GetArrayFromImage(img)).float()
        if img.GetNumberOfComponentsPerPixel() > 1:  # grab the first component
            arr = arr[:, :, :, 0]
        orig_idx = list(range(0, arr.shape[0], max(1, args.frame_stride)))
        if n_frames is not None and 0 < n_frames < len(orig_idx):
            # Deterministic per-path subset so the same frames are sampled each run.
            seed = (args.seed + zlib.crc32(path.encode())) & 0x7FFFFFFF
            g = torch.Generator().manual_seed(seed)
            sel = torch.randperm(len(orig_idx), generator=g)[:n_frames].sort().values
            orig_idx = [orig_idx[i] for i in sel.tolist()]
        arr = arr[orig_idx]
        arr = arr.unsqueeze(1)  # (T', 1, H, W)
        if bool(args.repeat_channel):
            arr = arr.repeat(1, 3, 1, 1)  # (T', 3, H, W)
        arr = transform(arr)
        return arr, orig_idx
    except Exception:
        print("Error reading cine: " + full, file=sys.stderr)
        return None, None


class BlindSweepFrameDataset(Dataset):
    """One item per cine: all frames (transformed) plus the file path and the
    original frame indices, so clustering can be reported per frame."""

    def __init__(self, df, args, transform):
        self.df = df
        self.args = args
        self.transform = transform

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        path = self.df.iloc[idx][self.args.img_column]
        n_frames = self.args.frames_per_sweep if self.args.frames_per_sweep > 0 else None
        frames, orig_idx = _read_cine(path, self.args, self.transform, n_frames=n_frames)
        if frames is None:
            return None
        return {"frames": frames, "file_path": path, "orig_idx": orig_idx}


@torch.no_grad()
def extract_embeddings(model, df, args, device, transform):
    """Embed cine frames. Returns ``(feats, records)`` where ``feats`` is
    ``(N, emb_dim)`` (CPU, L2-normalized) and ``records[i]`` is the
    ``(file_path, frame_index)`` of embedding ``i``. With ``args.n_samples > 0``
    stops once that many frames are collected (cines pre-shuffled in ``main``)."""
    ds = BlindSweepFrameDataset(df, args, transform)
    loader = DataLoader(
        ds,
        batch_size=args.cines_per_batch,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=lambda b: [x for x in b if x is not None],
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        worker_init_fn=_worker_init,
    )

    target = args.n_samples if args.n_samples > 0 else None
    embeddings, records = [], []
    n_cines = 0
    for batch in loader:
        for item in batch:
            frames = item["frames"]  # (T', 3, H, W)
            # Sub-batch the frames so a long cine never blows up GPU memory.
            zs = []
            for s in range(0, frames.shape[0], args.batch_size):
                chunk = frames[s:s + args.batch_size].to(device, non_blocking=True)
                zs.append(model(chunk).float().cpu())  # forward L2-normalizes
            embeddings.append(torch.cat(zs, dim=0))
            records.extend((item["file_path"], int(fi)) for fi in item["orig_idx"])
            n_cines += 1
        if n_cines and n_cines % 50 == 0:
            print(f"  embedded {n_cines} cines, {len(records)} frames", flush=True)
        if target is not None and len(records) >= target:
            break
    if not embeddings:
        raise SystemExit("No frames were embedded (all cines failed to read?).")
    feats = torch.cat(embeddings, dim=0)
    if target is not None and feats.shape[0] > target:  # trim the tail to exactly N
        feats = feats[:target]
        records = records[:target]
    return feats, records


def _kmeans(feats, k, num_init, device):
    """torch_kmeans on ``(N, D)`` feats -> ``(labels (N,), centers (k, D))`` with
    L2-normalized centers (matches SimNorth's prototype convention)."""
    x = feats.to(device).unsqueeze(0)  # (1, N, D)
    res = KMeans(n_clusters=k, num_init=num_init, seed=0, verbose=False)(x)
    labels = res.labels[0].cpu()
    centers = torch.nn.functional.normalize(res.centers[0], dim=1).cpu()
    return labels, centers


def silhouette_sweep(feats, args, device):
    """Sweep k over ``[min, max]`` (stride ``step``) on a subsample, returning
    ``(ks, scores)``. Subsampling bounds the O(N^2) silhouette cost."""
    sub = feats
    n = args.n_cluster_samples
    if 0 < n < feats.shape[0]:
        sub = feats[torch.randperm(feats.shape[0])[:n]]

    k_max = min(args.n_clusters_max, sub.shape[0] - 1)
    step = max(1, args.n_clusters_step)
    ks, scores = [], []
    for k in range(args.n_clusters_min, k_max + 1, step):
        labels, _ = _kmeans(sub, k, args.num_init, device)
        if torch.unique(labels).numel() < 2:
            continue
        s = SimNorth._silhouette_score(sub, labels)
        if torch.isnan(s):
            continue
        ks.append(k)
        scores.append(float(s))
        print(f"  k={k:3d}  silhouette={float(s):.4f}", flush=True)
    return ks, scores


def plot_silhouette(ks, scores, chosen_k, out_path):
    plt.figure(figsize=(7, 4))
    plt.plot(ks, scores, marker="o")
    if chosen_k in ks:
        plt.scatter([chosen_k], [scores[ks.index(chosen_k)]], color="red", zorder=5,
                    label=f"chosen k={chosen_k}")
        plt.legend()
    plt.xlabel("number of clusters (k)")
    plt.ylabel("mean silhouette")
    plt.title("Silhouette vs. k")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def representative_indices(feats, labels, centers, n_per_cluster):
    """For each cluster, the global embedding indices of the ``n_per_cluster``
    frames most cosine-similar to the centroid (descending)."""
    reps = {}
    for c in range(centers.shape[0]):
        member_idx = torch.nonzero(labels == c, as_tuple=False).flatten()
        if member_idx.numel() == 0:
            reps[c] = []
            continue
        sims = feats[member_idx] @ centers[c]  # cosine (both L2-normalized)
        order = torch.argsort(sims, descending=True)[:n_per_cluster]
        reps[c] = member_idx[order].tolist()
    return reps


def render_representatives(needed_idx, records, args, transform):
    """Read the (few) cines that hold representative frames and return
    ``{global_idx: (3, H, W) tensor}`` for display. Groups by file to read each
    cine at most once."""
    by_path = defaultdict(list)
    for gi in needed_idx:
        path, frame_idx = records[gi]
        by_path[path].append((gi, frame_idx))

    images = {}
    for path, items in by_path.items():
        frames, orig_idx = _read_cine(path, args, transform)
        if frames is None:
            continue
        pos = {oi: i for i, oi in enumerate(orig_idx)}
        for gi, frame_idx in items:
            if frame_idx in pos:
                images[gi] = frames[pos[frame_idx]]
    return images


def plot_cluster_grid(reps, images, out_path, n_per_cluster):
    """Montage: one row per cluster, its representative frames left-to-right."""
    clusters = sorted(reps)
    if not clusters:
        return
    fig, axes = plt.subplots(
        len(clusters), n_per_cluster,
        figsize=(1.6 * n_per_cluster, 1.6 * len(clusters)),
        squeeze=False,
    )
    for r, c in enumerate(clusters):
        idxs = reps[c]
        for col in range(n_per_cluster):
            ax = axes[r][col]
            ax.axis("off")
            if col < len(idxs) and idxs[col] in images:
                img = images[idxs[col]]  # (3, H, W) in [0, 1]
                ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
            if col == 0:
                ax.set_title(f"cluster {c} (n={len(idxs)})", fontsize=8, loc="left")
    fig.suptitle("Representative frames per cluster", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    transform = SimTestTransforms(args.img_size)

    print(f"Loading model from {args.model}")
    model = SimNorth.load_from_checkpoint(args.model, map_location=device)
    model.eval().to(device)

    df = _read_table(args.csv).reset_index(drop=True)
    print(f"Loaded {len(df)} cines from {args.csv}")

    if args.n_samples > 0:
        # Randomize sweep order so the first-N frames are an unbiased sample.
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        print(f"Sampling mode: shuffled sweeps, {args.frames_per_sweep} frames/sweep "
              f"until {args.n_samples} total frames")

    print("Extracting per-frame embeddings...")
    feats, records = extract_embeddings(model, df, args, device, transform)
    print(f"Embeddings: {tuple(feats.shape)} over {len(set(p for p, _ in records))} cines")

    print("Silhouette sweep...")
    ks, scores = silhouette_sweep(feats, args, device)
    if not ks:
        raise SystemExit("Silhouette sweep produced no valid k (too few frames or degenerate embeddings).")

    chosen_k = args.n_clusters if args.n_clusters and args.n_clusters > 0 else ks[int(np.argmax(scores))]
    print(f"Chosen k = {chosen_k}")
    plot_silhouette(ks, scores, chosen_k, os.path.join(args.out, "silhouette.png"))

    print(f"Final clustering at k={chosen_k} over all {feats.shape[0]} frames...")
    labels, centers = _kmeans(feats, chosen_k, args.num_init, device)

    # Per-frame output: file_path, index (frame index in the cine), cluster_label
    out_df = pd.DataFrame({
        "file_path": [p for p, _ in records],
        "index": [fi for _, fi in records],
        "cluster_label": labels.numpy(),
    })
    csv_path = os.path.join(args.out, "clusters.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(out_df)} frames)")

    reps = representative_indices(feats, labels, centers, args.grid_n)
    needed = [gi for lst in reps.values() for gi in lst]
    images = render_representatives(needed, records, args, transform)
    plot_cluster_grid(reps, images, os.path.join(args.out, "clusters_grid.png"), args.grid_n)
    print(f"Wrote {os.path.join(args.out, 'clusters_grid.png')}")

    sizes = {int(c): int((labels == c).sum()) for c in range(chosen_k)}
    summary = {
        "model": args.model,
        "csv": args.csv,
        "n_cines": len(set(p for p, _ in records)),
        "n_frames": int(feats.shape[0]),
        "chosen_k": int(chosen_k),
        "silhouette_at_chosen_k": float(scores[ks.index(chosen_k)]) if chosen_k in ks else None,
        "sweep": {"k": ks, "silhouette": scores},
        "cluster_sizes": sizes,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Outputs in {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster-evaluate a trained SimNorth model over blind-sweep cines")

    io_group = parser.add_argument_group("I/O")
    io_group.add_argument("--model", required=True, type=str, help="Path to the SimNorth checkpoint (.ckpt)")
    io_group.add_argument("--csv", required=True, type=str, help="CSV/parquet of blind-sweep cines")
    io_group.add_argument("--mount_point", default="./", type=str, help="Dataset mount directory")
    io_group.add_argument("--img_column", default="file_path", type=str, help="Cine path column in the table")
    io_group.add_argument("--out", default="./eval_out", type=str, help="Output directory")

    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--img_size", default=256, type=int, help="Square center-crop size (match training)")
    data_group.add_argument("--frame_stride", default=1, type=int, help="Keep every Nth frame per cine (1 = all frames)")
    data_group.add_argument("--n_samples", default=0, type=int, help="Total frames to sample for clustering (0 = use every frame of every cine)")
    data_group.add_argument("--frames_per_sweep", default=16, type=int, help="Random frames to take per cine when --n_samples > 0")
    data_group.add_argument("--seed", default=0, type=int, help="Seed for sweep shuffling and per-cine frame sampling")
    data_group.add_argument("--batch_size", default=256, type=int, help="Frames per forward pass")
    data_group.add_argument("--cines_per_batch", default=1, type=int, help="Cines read per dataloader batch")
    data_group.add_argument("--num_workers", default=4, type=int, help="Dataloader workers (cine readers)")
    data_group.add_argument("--prefetch_factor", default=2, type=int, help="Dataloader prefetch factor")
    data_group.add_argument("--repeat_channel", default=1, type=int, help="Repeat grayscale frames to 3 channels")

    cl_group = parser.add_argument_group("Clustering")
    cl_group.add_argument("--n_clusters_min", default=2, type=int, help="Minimum k for the silhouette sweep")
    cl_group.add_argument("--n_clusters_max", default=40, type=int, help="Maximum k for the silhouette sweep")
    cl_group.add_argument("--n_clusters_step", default=1, type=int, help="Stride for the k sweep")
    cl_group.add_argument("--n_clusters", default=0, type=int, help="Force this k for assignment/grids (0 = silhouette-best)")
    cl_group.add_argument("--n_cluster_samples", default=4096, type=int, help="Subsample size for the O(N^2) silhouette (<=0 uses all)")
    cl_group.add_argument("--num_init", default=10, type=int, help="KMeans restarts (num_init)")
    cl_group.add_argument("--grid_n", default=8, type=int, help="Representative frames per cluster in the montage")

    parser.add_argument("--device", default=None, type=str, help="torch device (default: cuda if available)")

    main(parser.parse_args())

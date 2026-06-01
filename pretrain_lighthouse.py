"""Optimize light house anchors and pickle them for SimNorth training.

Example:

    python pretrain_lighthouse.py --n_lights 64 --emb_dim 128 --out lights.pkl

Then pass them to training:

    python train.py ... --lights lights.pkl
"""

import argparse
import pickle

import torch
from lightning.pytorch import Trainer

from simnorth import LightHouse


def main(args):
    model = LightHouse(
        n_lights=args.n_lights,
        emb_dim=args.emb_dim,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        n_iter=args.n_iter,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model)

    lights = torch.abs(torch.nn.functional.normalize(model.light_house.detach())).cpu().numpy()
    with open(args.out, "wb") as f:
        pickle.dump(lights, f)
    print(f"Saved {lights.shape} light house anchors to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain light house anchors")
    parser.add_argument("--n_lights", default=64, type=int, help="Number of anchors")
    parser.add_argument("--emb_dim", default=128, type=int, help="Embedding dimension")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay")
    parser.add_argument("--epochs", default=10, type=int, help="Max epochs")
    parser.add_argument("--n_iter", default=1000, type=int, help="Optimization steps per epoch")
    parser.add_argument("--out", default="lights.pkl", type=str, help="Output pickle path")
    main(parser.parse_args())

#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hetero_rlhf.data import PairEmbeddingDataset
from hetero_rlhf.models import PersonalizedRewardModel
from hetero_rlhf.transfer import NewUserRewardHead, transfer_step
from hetero_rlhf.evaluation import preference_accuracy_from_rewards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    ckpt = torch.load(args.base_checkpoint, map_location="cpu")
    base = PersonalizedRewardModel(
        ckpt["input_dim"], ckpt["representation_dim"], ckpt["num_users"],
        ckpt["hidden_dim"], ckpt["representation"],
    )
    base.load_state_dict(ckpt["state_dict"])

    model = NewUserRewardHead(base.representation, ckpt["representation_dim"])
    optimizer = torch.optim.Adam([model.theta], lr=args.lr)

    train = PairEmbeddingDataset(args.train)
    val = PairEmbeddingDataset(args.val)
    for epoch in range(args.epochs):
        loss = transfer_step(model, train.chosen, train.rejected, optimizer)
        with torch.no_grad():
            rc = model.reward(val.chosen).numpy()
            rr = model.reward(val.rejected).numpy()
        acc = preference_accuracy_from_rewards(rc, rr)
        print(f"epoch={epoch+1} loss={loss:.4f} val_accuracy={acc:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"theta": model.theta.detach(), "base_checkpoint": args.base_checkpoint}, args.output)


if __name__ == "__main__":
    main()

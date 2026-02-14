#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hetero_rlhf.data import PairEmbeddingDataset
from hetero_rlhf.models import PersonalizedRewardModel
from hetero_rlhf.personalized import train_personalized_epoch, preference_accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--representation", choices=["general", "linear"], default="general")
    p.add_argument("--representation-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    train_ds = PairEmbeddingDataset(args.train)
    val_ds = PairEmbeddingDataset(args.val)
    input_dim = train_ds.chosen.shape[1]
    num_users = int(train_ds.user_id.max().item()) + 1

    model = PersonalizedRewardModel(
        input_dim=input_dim,
        representation_dim=args.representation_dim,
        num_users=num_users,
        hidden_dim=args.hidden_dim,
        representation=args.representation,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        loss = train_personalized_epoch(model, loader, optimizer)
        acc = preference_accuracy(
            model, val_ds.chosen, val_ds.rejected, val_ds.user_id
        )
        print(f"epoch={epoch+1} loss={loss:.4f} val_accuracy={acc:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "representation_dim": args.representation_dim,
        "num_users": num_users,
        "hidden_dim": args.hidden_dim,
        "representation": args.representation,
    }, args.output)


if __name__ == "__main__":
    main()

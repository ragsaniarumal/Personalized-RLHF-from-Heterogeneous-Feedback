from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def pairwise_btl_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
) -> torch.Tensor:
    """Negative Bradley-Terry log likelihood for chosen > rejected."""
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


@torch.no_grad()
def preference_accuracy(
    model: nn.Module,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    user_ids: torch.Tensor | None = None,
) -> float:
    if user_ids is None:
        rc = model.reward(chosen)
        rr = model.reward(rejected)
    else:
        rc = model.reward(chosen, user_ids)
        rr = model.reward(rejected, user_ids)
    return float((rc > rr).float().mean().item())


def train_personalized_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
) -> float:
    model.train()
    losses = []
    for batch in loader:
        chosen = batch["chosen"].to(device)
        rejected = batch["rejected"].to(device)
        user_ids = batch["user_id"].to(device)
        rc, rr = model.pair_rewards(chosen, rejected, user_ids)
        loss = pairwise_btl_loss(rc, rr)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / max(len(losses), 1)

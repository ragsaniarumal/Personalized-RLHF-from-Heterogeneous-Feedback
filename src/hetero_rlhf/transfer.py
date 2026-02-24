from __future__ import annotations

import torch
from torch import nn

from .personalized import pairwise_btl_loss


class NewUserRewardHead(nn.Module):
    """Algorithm 2: frozen shared representation plus one new theta_0."""

    def __init__(self, frozen_representation: nn.Module, representation_dim: int):
        super().__init__()
        self.representation = frozen_representation
        for parameter in self.representation.parameters():
            parameter.requires_grad_(False)
        self.theta = nn.Parameter(torch.zeros(representation_dim))
        nn.init.normal_(self.theta, mean=0.0, std=0.02)

    def reward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.representation(x)
        return (z * self.theta).sum(dim=-1)


def transfer_step(
    model: NewUserRewardHead,
    chosen: torch.Tensor,
    rejected: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    rc = model.reward(chosen)
    rr = model.reward(rejected)
    loss = pairwise_btl_loss(rc, rr)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())

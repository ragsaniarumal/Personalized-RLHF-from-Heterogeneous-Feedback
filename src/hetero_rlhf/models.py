from __future__ import annotations

import torch
from torch import nn


class RepresentationHead(nn.Module):
    """Shared representation psi_omega used by every annotator reward model."""

    def __init__(
        self,
        input_dim: int,
        representation_dim: int,
        hidden_dim: int = 512,
        kind: str = "general",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(input_dim, representation_dim, bias=False)
        elif kind == "general":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, representation_dim),
                nn.LayerNorm(representation_dim),
            )
        else:
            raise ValueError("kind must be 'general' or 'linear'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PersonalizedRewardModel(nn.Module):
    """Shared representation with a separate theta_i vector for each user."""

    def __init__(
        self,
        input_dim: int,
        representation_dim: int,
        num_users: int,
        hidden_dim: int = 512,
        representation: str = "general",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.representation = RepresentationHead(
            input_dim=input_dim,
            representation_dim=representation_dim,
            hidden_dim=hidden_dim,
            kind=representation,
            dropout=dropout,
        )
        self.user_vectors = nn.Embedding(num_users, representation_dim)
        nn.init.normal_(self.user_vectors.weight, mean=0.0, std=0.02)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.representation(x)

    def reward(self, x: torch.Tensor, user_ids: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        theta = self.user_vectors(user_ids)
        return (z * theta).sum(dim=-1)

    def pair_rewards(
        self,
        chosen: torch.Tensor,
        rejected: torch.Tensor,
        user_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.reward(chosen, user_ids), self.reward(rejected, user_ids)


class PooledRewardModel(nn.Module):
    """Naive baseline: all annotators share one reward vector."""

    def __init__(
        self,
        input_dim: int,
        representation_dim: int,
        hidden_dim: int = 512,
        representation: str = "general",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.representation = RepresentationHead(
            input_dim,
            representation_dim,
            hidden_dim,
            representation,
            dropout,
        )
        self.theta = nn.Parameter(torch.zeros(representation_dim))
        nn.init.normal_(self.theta, mean=0.0, std=0.02)

    def reward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.representation(x) * self.theta).sum(dim=-1)

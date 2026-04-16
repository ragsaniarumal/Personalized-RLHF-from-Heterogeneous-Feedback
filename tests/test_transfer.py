import torch

from hetero_rlhf.models import RepresentationHead
from hetero_rlhf.transfer import NewUserRewardHead


def test_transfer_freezes_representation():
    rep = RepresentationHead(8, 3, hidden_dim=6, kind="general", dropout=0.0)
    model = NewUserRewardHead(rep, 3)
    assert model.theta.requires_grad
    assert all(not p.requires_grad for p in model.representation.parameters())

import torch

from hetero_rlhf.models import PersonalizedRewardModel
from hetero_rlhf.personalized import pairwise_btl_loss


def test_personalized_reward_shapes():
    model = PersonalizedRewardModel(
        input_dim=16,
        representation_dim=4,
        num_users=3,
        hidden_dim=8,
        representation="general",
        dropout=0.0,
    )
    x = torch.randn(5, 16)
    users = torch.tensor([0, 1, 2, 0, 1])
    reward = model.reward(x, users)
    assert reward.shape == (5,)


def test_btl_loss_prefers_larger_margin():
    good = pairwise_btl_loss(torch.tensor([2.0]), torch.tensor([0.0]))
    bad = pairwise_btl_loss(torch.tensor([0.0]), torch.tensor([2.0]))
    assert good < bad

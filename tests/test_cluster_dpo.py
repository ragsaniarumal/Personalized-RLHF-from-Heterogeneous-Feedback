import numpy as np
import torch

from hetero_rlhf.cluster_dpo import (
    dpo_log_preference,
    assign_users_from_logprobs,
    cluster_members,
)


def test_dpo_score_rewards_preferred_margin():
    score_good = dpo_log_preference(
        torch.tensor([3.0]), torch.tensor([1.0]),
        torch.tensor([1.0]), torch.tensor([1.0]), beta=1.0
    )
    score_bad = dpo_log_preference(
        torch.tensor([1.0]), torch.tensor([3.0]),
        torch.tensor([1.0]), torch.tensor([1.0]), beta=1.0
    )
    assert score_good > score_bad


def test_em_assignment():
    users = np.array([0, 0, 1, 1])
    scores = np.array([
        [-0.1, -0.1, -2.0, -2.0],
        [-2.0, -2.0, -0.1, -0.1],
    ])
    assignment = assign_users_from_logprobs(users, scores)
    assert assignment == {0: 0, 1: 1}
    assert cluster_members(assignment, 2) == {0: [0], 1: [1]}

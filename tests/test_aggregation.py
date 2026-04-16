import numpy as np

from hetero_rlhf.aggregation import aggregate_rewards


def test_alpha_zero_is_mean():
    rewards = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 2.0]])
    np.testing.assert_allclose(aggregate_rewards(rewards, 0.0), rewards.mean(axis=1))


def test_extreme_alpha_moves_toward_extrema():
    rewards = np.array([[1.0, 2.0, 5.0]])
    negative = aggregate_rewards(rewards, -50.0)[0]
    positive = aggregate_rewards(rewards, 50.0)[0]
    assert negative < rewards.mean()
    assert positive > rewards.mean()
    assert abs(negative - 1.0) < 0.1
    assert abs(positive - 5.0) < 0.1

from __future__ import annotations

import math
import numpy as np
import torch


def aggregate_rewards(rewards, alpha: float, axis: int = -1):
    """Numerically stable implementation of Park et al.'s Agg_alpha.

    `rewards` may be a NumPy array or torch.Tensor. The aggregation axis
    corresponds to users.
    """
    if isinstance(rewards, torch.Tensor):
        if abs(alpha) < 1e-12:
            return rewards.mean(dim=axis)
        scaled = alpha * rewards
        n = rewards.shape[axis]
        return (torch.logsumexp(scaled, dim=axis) - math.log(n)) / alpha

    arr = np.asarray(rewards, dtype=np.float64)
    if abs(alpha) < 1e-12:
        return arr.mean(axis=axis)
    scaled = alpha * arr
    m = np.max(scaled, axis=axis, keepdims=True)
    lme = np.log(np.mean(np.exp(scaled - m), axis=axis)) + np.squeeze(m, axis=axis)
    return lme / alpha


def sweep_aggregation(reward_matrix, alphas):
    return {float(alpha): aggregate_rewards(reward_matrix, float(alpha), axis=-1)
            for alpha in alphas}

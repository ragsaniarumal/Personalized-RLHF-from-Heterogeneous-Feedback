from __future__ import annotations

import numpy as np


def preference_accuracy_from_rewards(chosen_rewards, rejected_rewards) -> float:
    chosen = np.asarray(chosen_rewards)
    rejected = np.asarray(rejected_rewards)
    if chosen.shape != rejected.shape:
        raise ValueError("chosen and rejected rewards must have the same shape")
    return float(np.mean(chosen > rejected))


def per_user_accuracy(user_ids, chosen_rewards, rejected_rewards) -> dict[int, float]:
    users = np.asarray(user_ids)
    chosen = np.asarray(chosen_rewards)
    rejected = np.asarray(rejected_rewards)
    return {
        int(user): float(np.mean(chosen[users == user] > rejected[users == user]))
        for user in np.unique(users)
    }

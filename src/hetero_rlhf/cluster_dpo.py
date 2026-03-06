from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def dpo_log_preference(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Log probability term used by the ClusterDPO E-step."""
    margin = beta * (
        (chosen_logp - ref_chosen_logp)
        - (rejected_logp - ref_rejected_logp)
    )
    return F.logsigmoid(margin)


def assign_users_from_logprobs(
    user_ids,
    cluster_log_scores,
) -> dict[int, int]:
    """Assign every user to the cluster with maximum summed log score.

    Parameters
    ----------
    user_ids:
        One user id per preference pair, shape [n_pairs].
    cluster_log_scores:
        Pairwise log preference scores, shape [n_clusters, n_pairs].
    """
    users = np.asarray(user_ids)
    scores = np.asarray(cluster_log_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != users.shape[0]:
        raise ValueError("cluster_log_scores must have shape [K, n_pairs]")

    assignments = {}
    for user in np.unique(users):
        mask = users == user
        per_cluster = scores[:, mask].sum(axis=1)
        assignments[int(user)] = int(np.argmax(per_cluster))
    return assignments


def cluster_members(assignments: dict[int, int], k: int) -> dict[int, list[int]]:
    return {
        cluster: sorted(user for user, assigned in assignments.items() if assigned == cluster)
        for cluster in range(k)
    }

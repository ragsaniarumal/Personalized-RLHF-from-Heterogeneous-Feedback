#!/usr/bin/env python
"""Small synthetic sanity check for the ClusterDPO E-step."""
import numpy as np
from hetero_rlhf.cluster_dpo import assign_users_from_logprobs, cluster_members

user_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
scores = np.array([
    [-0.1, -0.2, -0.2, -0.1, -2.0, -1.5, -1.2, -1.4],
    [-1.8, -1.5, -1.3, -1.4, -0.1, -0.2, -0.2, -0.1],
])
assignments = assign_users_from_logprobs(user_ids, scores)
print(assignments)
print(cluster_members(assignments, k=2))

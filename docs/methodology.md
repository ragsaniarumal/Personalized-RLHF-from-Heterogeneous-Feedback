# Methodology

## Problem

The project follows Park et al. (2024) and treats preference heterogeneity as a modelling problem rather than annotation noise.

For user `i` and a response/trajectory `tau`:

```text
r_i(tau) = <psi_omega(phi(tau)), theta_i>
```

`phi` is a frozen language-model representation, `psi_omega` is learned jointly from all users, and `theta_i` captures user-specific preferences.

For a chosen response `tau_c` and rejected response `tau_r`, the Bradley-Terry likelihood is

```text
P(tau_c > tau_r) = sigmoid(r_i(tau_c) - r_i(tau_r)).
```

## Algorithm 1

`PersonalizedRewardModel` implements a common representation module plus an embedding table of user reward vectors. The training loss is pairwise negative log likelihood.

The `general` variant uses a two-layer nonlinear representation head. The `linear` variant replaces it with one linear map to reproduce the project's general-vs-linear representation comparison.

## Algorithm 2

`NewUserRewardHead` receives the trained representation from Algorithm 1 and freezes every shared parameter. Only a fresh user vector is optimized.

This makes the transfer experiment measurable as preference accuracy versus the number of new-user comparisons.

## Algorithm 3

The project presentation described an EM-style ClusterDPO procedure.

E-step:

```text
assign user i to argmax_k sum_j log sigma(
    beta * [
      log pi_k(y_w|x) - log pi_ref(y_w|x)
      - log pi_k(y_l|x) + log pi_ref(y_l|x)
    ]
)
```

M-step:

```text
update policy k with DPO on preference pairs from users assigned to cluster k.
```

The core E-step is implemented in `hetero_rlhf.cluster_dpo` and can be unit tested from precomputed log-probabilities. The heavyweight policy-update path is isolated in `scripts/cluster_dpo_train.py`.

## Algorithm 4

For a vector of per-user rewards `r`, the aggregation family is

```text
Agg_alpha(r) =
  log(mean(exp(alpha * r_i))) / alpha, alpha != 0
  mean(r_i),                         alpha = 0
```

A numerically stable log-sum-exp implementation is used.

As `alpha` moves negative, the aggregate increasingly protects low-reward users. As it moves positive, it increasingly tracks high-reward users.

## Evaluation

The historical course metric is pairwise preference accuracy:

```text
mean(r(chosen) > r(rejected))
```

Per-user accuracy should also be reported so a population average does not hide minority-user failures.

## Reproduction guidance

The exact GPT-J 6B and LLaMA3 8B runs are expensive. The repository separates the algorithmic core from the backbone encoder so development can be done with smaller models while retaining the same objective.

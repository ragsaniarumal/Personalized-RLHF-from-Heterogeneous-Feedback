# Historical Results and Interpretation

The original course repository and checkpoints were lost. The table below is transcribed from the retained course presentation and is stored verbatim in `experiments/historical_results.csv`.

| Method | GPT-J 6B — Original | GPT-J 6B — Course | LLaMA3 8B — Original | LLaMA3 8B — Course |
|---|---:|---:|---:|---:|
| Naive | 0.655 | 0.691 | 0.660 | 0.572 |
| PG | 0.670 | 0.607 | 0.685 | 0.690 |
| PL | 0.658 | 0.695 | 0.535 | 0.508 |
| CG | 0.668 | 0.611 | 0.682 | 0.743 |
| CL | 0.530 | 0.622 | 0.525 | 0.391 |

## What the table supports

- `CG` is the strongest recorded LLaMA3 course result at **0.743**.
- `PG` substantially exceeds the recorded LLaMA3 Naive run: `0.690` versus `0.572`.
- Linear representation variants are weak on the recorded LLaMA3 experiments.
- GPT-J is mixed: PL slightly exceeds Naive, while PG and CG do not.

## Transfer result retained from the presentation

Algorithm 2 was reported as staying within approximately one percentage point of Algorithm 1 on held-out users while learning only the new user parameter on top of the frozen representation.

The per-shot curve and raw checkpoint are no longer available, so the reconstructed repository does not manufacture them.

## Cluster sweep retained from the presentation

The retained slides report:

```text
K=1 (high bias) < K=2 (best) > K=5 (high variance)
```

This was used as an empirical illustration of the bias-variance trade-off discussed in the paper.

## Aggregation result retained from the presentation

The project qualitatively observed:

- negative `alpha`: more conservative / least-satisfied-user-aware behavior;
- `alpha=0`: balanced average-preference behavior;
- positive `alpha`: more assertive behavior reflecting high-reward users.

Because the original generated summaries are unavailable, this qualitative observation is documented but not recreated as a fake artifact.

## Slide inconsistency

One slide states that PG exceeds Naive by approximately 1.5–2.5 percentage points on LLaMA3. The table on the same presentation gives:

```text
course: 0.690 - 0.572 = 0.118
paper:  0.685 - 0.660 = 0.025
```

The 2.5-point statement is consistent with the Original-column numbers, not with the course-column numbers. This repository therefore reports the table directly and does not repeat that caption as a course-run claim.

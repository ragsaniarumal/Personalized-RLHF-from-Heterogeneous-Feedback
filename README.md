# Personalized RLHF from Heterogeneous Feedback

Course project by **Tanmay Gejapati and Rumal Ragsania**.

A clean reconstruction of a CS-6103 course project that implemented and empirically studied the four main algorithmic ideas from **Park et al. (2024), _RLHF from Heterogeneous Feedback via Personalization and Preference Aggregation_**.

The project asks a simple question:

> What should RLHF do when different annotators genuinely prefer different kinds of responses?

Instead of forcing all feedback into a single reward model, the implementation studies four alternatives:

1. **Personalized reward learning** with a shared representation and user-specific reward heads.
2. **Transfer to a new user** by freezing the shared representation and learning only that user's reward vector.
3. **ClusterDPO** with an EM-style assignment/update loop and one DPO policy per preference cluster.
4. **Reward aggregation** with the paper's social-choice-inspired `alpha` family.

The original course repository is no longer available. This repository reconstructs the implementation from the retained project presentation, paper, and experiment records. Historical numbers are therefore kept separately from any newly reproduced runs.

## Course setup retained from the original project

- **Dataset:** Reddit TL;DR preference comparisons from `openai/summarize_from_feedback`
- **Annotators:** top 5 workers by comparison count
- **Training balance:** 5,373 comparisons per worker
- **Historical training pairs:** 26,865
- **Historical validation pairs:** 6,190
- **Backbones evaluated:** GPT-J 6B and LLaMA 3 8B
- **Metric:** pairwise preference accuracy

## Historical course results

These are the values recorded in the retained course presentation.

| Method | GPT-J 6B — Original | GPT-J 6B — Course | LLaMA3 8B — Original | LLaMA3 8B — Course |
|---|---:|---:|---:|---:|
| Naive | 0.655 | **0.691** | 0.660 | 0.572 |
| PG | 0.670 | 0.607 | 0.685 | **0.690** |
| PL | 0.658 | **0.695** | 0.535 | 0.508 |
| CG | 0.668 | 0.611 | 0.682 | **0.743** |
| CL | 0.530 | 0.622 | 0.525 | 0.391 |

Tags used in the presentation:

- `Naive` — one shared model
- `PG` — personalized model with a general learned representation
- `PL` — personalized model with a linear representation
- `CG` — clustered model (`K=2`) with a general representation
- `CL` — clustered model (`K=2`) with a linear representation

The strongest historical result was **0.743 preference accuracy for CG on LLaMA3 8B**. The retained slides also report that transfer to held-out users stayed within roughly one percentage point of Algorithm 1 and that `K=2` performed better than `K=1` and `K=5` in the ClusterDPO sweep.

Because the original logs and checkpoints were lost, these historical values are not presented as outputs of the reconstructed code. See [`docs/results.md`](docs/results.md).

## Algorithm 1 — personalized reward learning

For frozen response features `phi(tau)`, the model learns a shared representation `psi_omega` and a separate reward vector `theta_i` for each annotator:

```text
frozen LM embedding
        |
        v
 shared representation
        |
        +---- theta_user_1
        +---- theta_user_2
        +---- ...
```

The reward is

```text
r_i(tau) = < psi_omega(phi(tau)), theta_i >
```

and pairwise preferences are trained with a Bradley-Terry / logistic objective.

Two representation choices are included:

- `general`: two-layer learned MLP
- `linear`: single learned linear projection

## Algorithm 2 — transfer to a new user

After Algorithm 1, `psi_omega` is frozen. A new annotator only needs a fresh `theta_0`.

This isolates the central transfer hypothesis: expensive shared representation learning can be reused while a new user is personalized from comparatively little feedback.

## Algorithm 3 — ClusterDPO

ClusterDPO alternates between:

- **E-step:** assign each user to the policy that best explains their pairwise preferences under the DPO objective.
- **M-step:** update each cluster policy using preference pairs from users currently assigned to that cluster.

The repository separates the cluster-assignment mathematics from the heavyweight LLM training backend. The core E-step can be tested from precomputed chosen/rejected log-probabilities, while [`scripts/cluster_dpo_train.py`](scripts/cluster_dpo_train.py) contains the optional TRL/PEFT training path.

## Algorithm 4 — reward aggregation

Individual user rewards are combined using

```text
Agg_alpha(r) = log(mean(exp(alpha * r_i))) / alpha,   alpha != 0
Agg_0(r)     = mean(r_i)
```

Interpretation:

- very negative `alpha` emphasizes the least-satisfied users;
- `alpha = 0` is utilitarian averaging;
- very positive `alpha` emphasizes the highest rewards.

[`scripts/run_aggregation.py`](scripts/run_aggregation.py) sweeps this fairness parameter on stored reward matrices.

## Repository layout

```text
.
├── configs/
│   └── base.json
├── docs/
│   ├── methodology.md
│   └── results.md
├── experiments/
│   ├── historical_results.csv
│   └── historical_run_metadata.json
├── notebooks/
│   └── course_results_analysis.ipynb
├── scripts/
│   ├── prepare_data.py
│   ├── embed_pairs.py
│   ├── train_personalized.py
│   ├── train_transfer.py
│   ├── cluster_dpo_train.py
│   └── run_aggregation.py
├── src/hetero_rlhf/
│   ├── aggregation.py
│   ├── cluster_dpo.py
│   ├── data.py
│   ├── evaluation.py
│   ├── models.py
│   ├── personalized.py
│   └── transfer.py
└── tests/
```

## Installation

The core algorithms and tests need only the lightweight dependencies:

```bash
python -m pip install -e .
pytest -q
```

For dataset download and large-model experiments:

```bash
python -m pip install -e ".[llm]"
```

## Reconstruct the preference dataset

```bash
python scripts/prepare_data.py \
    --output-dir data/processed \
    --top-users 5 \
    --train-per-user 5373
```

The loader is written against the `comparisons` portion of `openai/summarize_from_feedback`, where annotators choose between two summaries.

## Precompute frozen LM embeddings

```bash
python scripts/embed_pairs.py \
    --input data/processed/train.jsonl \
    --output data/processed/train_embeddings.npz \
    --workers-file data/processed/workers.txt \
    --model EleutherAI/gpt-j-6B
```

A smaller causal LM can be supplied while testing the pipeline.

## Train Algorithm 1

```bash
python scripts/train_personalized.py \
    --train data/processed/train_embeddings.npz \
    --val data/processed/val_embeddings.npz \
    --representation general \
    --output checkpoints/personalized_general.pt
```

Use `--representation linear` for the PL variant.

## Train Algorithm 2

```bash
python scripts/train_transfer.py \
    --base-checkpoint checkpoints/personalized_general.pt \
    --train data/processed/new_user_train_embeddings.npz \
    --val data/processed/new_user_val_embeddings.npz \
    --output checkpoints/new_user_theta.pt
```

## Run the aggregation sweep

```bash
python scripts/run_aggregation.py \
    --rewards path/to/reward_matrix.npy \
    --alphas -10 -2 0 2 10
```

## ClusterDPO

The E-step and its objective are implemented independently of any particular LLM stack. For actual policy updates, the optional backend uses Hugging Face TRL's `DPOTrainer`.

```bash
python scripts/cluster_dpo_train.py --help
```

Large-model reproduction requires substantial GPU memory and is intentionally not part of the automated test workflow.

## Reproducibility boundary

This repository distinguishes two things:

**Reconstructed implementation**
- the four algorithms;
- dataset preparation;
- representation/reward training;
- transfer;
- cluster assignment;
- aggregation;
- evaluation utilities;
- tests.

**Historical evidence**
- GPT-J/LLaMA3 result table;
- top-five-worker experimental setup;
- reported transfer result;
- reported `K=1,2,5` cluster sweep.

The latter came from the retained course presentation and is not silently regenerated.

## Reference

Chanwoo Park, Mingyang Liu, Dingwen Kong, Kaiqing Zhang, and Asuman Ozdaglar.  
**RLHF from Heterogeneous Feedback via Personalization and Preference Aggregation.** 2024.  
arXiv:2405.00254.

The summarization preference data originates from the OpenAI `summarize_from_feedback` release associated with _Learning to Summarize from Human Feedback_.

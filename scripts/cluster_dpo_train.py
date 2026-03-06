#!/usr/bin/env python
"""EM-style ClusterDPO training.

This is the heavyweight reproduction path. It trains one DPO policy per
cluster, then performs the E-step by rescoring every user's preference pairs
under every cluster policy and the fixed reference model.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from hetero_rlhf.cluster_dpo import assign_users_from_logprobs


def completion_logprob(model, tokenizer, prompt: str, completion: str, max_length: int) -> float:
    """Sum log p(completion tokens | prompt) for a causal LM."""
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length // 2,
    )["input_ids"]
    full = tokenizer(
        prompt + completion,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    device = next(model.parameters()).device
    input_ids = full["input_ids"].to(device)
    attention_mask = full["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    labels = input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # Token 0 predicts token 1. Completion begins after len(prompt_ids).
    start = max(len(prompt_ids) - 1, 0)
    return float(logp[:, start:].sum().detach().cpu())


def preference_logscore(
    policy,
    reference,
    tokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    beta: float,
    max_length: int,
) -> float:
    pc = completion_logprob(policy, tokenizer, prompt, chosen, max_length)
    pr = completion_logprob(policy, tokenizer, prompt, rejected, max_length)
    rc = completion_logprob(reference, tokenizer, prompt, chosen, max_length)
    rr = completion_logprob(reference, tokenizer, prompt, rejected, max_length)
    margin = beta * ((pc - rc) - (pr - rr))
    return float(torch.nn.functional.logsigmoid(torch.tensor(margin)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, help="JSON/JSONL with prompt/chosen/rejected/user_id")
    p.add_argument("--clusters", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--em-rounds", type=int, default=3)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--output-dir", default="checkpoints/cluster_dpo")
    args = p.parse_args()

    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
        if args.use_lora:
            from peft import LoraConfig
    except ImportError as exc:
        raise SystemExit('Install the LLM extras: pip install -e ".[llm]"') from exc

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    required = {"prompt", "chosen", "rejected", "user_id"}
    if not required.issubset(dataset.column_names):
        raise SystemExit(f"dataset must contain {sorted(required)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reference = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    reference.eval()

    users = sorted(set(int(x) for x in dataset["user_id"]))
    assignment = {u: (idx % args.clusters) for idx, u in enumerate(users)}
    policies = [None] * args.clusters

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for em_round in range(args.em_rounds):
        print(f"\nEM round {em_round + 1}/{args.em_rounds}")

        # M-step
        for cluster in range(args.clusters):
            member_users = {u for u, c in assignment.items() if c == cluster}
            subset = dataset.filter(lambda row: int(row["user_id"]) in member_users)
            if len(subset) == 0:
                print(f"cluster {cluster}: empty; skipping")
                continue

            model = policies[cluster]
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model, torch_dtype="auto", device_map="auto"
                )

            config = DPOConfig(
                output_dir=f"{args.output_dir}/round_{em_round}/cluster_{cluster}",
                beta=args.beta,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                num_train_epochs=1,
                logging_steps=10,
                save_strategy="no",
                max_length=args.max_length,
            )

            peft_config = None
            if args.use_lora:
                peft_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    task_type="CAUSAL_LM",
                )

            trainer = DPOTrainer(
                model=model,
                ref_model=reference if not args.use_lora else None,
                args=config,
                train_dataset=subset.remove_columns(["user_id"]),
                processing_class=tokenizer,
                peft_config=peft_config,
            )
            trainer.train()
            policies[cluster] = trainer.model
            trainer.save_model(f"{args.output_dir}/round_{em_round}/cluster_{cluster}")

        # No E-step needed after the final M-step.
        if em_round + 1 == args.em_rounds:
            break

        # E-step
        scores = np.zeros((args.clusters, len(dataset)), dtype=np.float64)
        for cluster, policy in enumerate(policies):
            if policy is None:
                scores[cluster, :] = -np.inf
                continue
            policy.eval()
            for idx, row in enumerate(dataset):
                scores[cluster, idx] = preference_logscore(
                    policy,
                    reference,
                    tokenizer,
                    row["prompt"],
                    row["chosen"],
                    row["rejected"],
                    args.beta,
                    args.max_length,
                )

        assignment = assign_users_from_logprobs(np.asarray(dataset["user_id"]), scores)
        print("E-step assignment:", assignment)

    print("\nFinal assignment:", assignment)


if __name__ == "__main__":
    main()

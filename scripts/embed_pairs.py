#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--workers-file", help="Optional fixed worker ordering from prepare_data.py")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    args = p.parse_args()

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('Install the LLM extras: pip install -e ".[llm]"') from exc

    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()]
    if args.workers_file:
        workers = [
            line.strip()
            for line in Path(args.workers_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        workers = sorted({r["worker"] for r in rows})
    worker_map = {w: i for i, w in enumerate(workers)}
    unknown = sorted({r["worker"] for r in rows} - set(worker_map))
    if unknown:
        raise SystemExit(f"input contains workers missing from worker map: {unknown}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    model.eval()
    device = next(model.parameters()).device

    chosen_all, rejected_all = [], []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        for key, target in (("chosen", chosen_all), ("rejected", rejected_all)):
            texts = [r["prompt"] + "\n\nSummary:\n" + r[key] for r in batch]
            tokens = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            with torch.no_grad():
                output = model(**tokens)
                pooled = mean_pool(output.last_hidden_state, tokens["attention_mask"])
            target.append(pooled.float().cpu().numpy())

    np.savez_compressed(
        args.output,
        chosen=np.concatenate(chosen_all),
        rejected=np.concatenate(rejected_all),
        user_id=np.array([worker_map[r["worker"]] for r in rows], dtype=np.int64),
        workers=np.array(workers),
    )


if __name__ == "__main__":
    main()

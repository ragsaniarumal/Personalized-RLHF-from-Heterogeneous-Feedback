#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from hetero_rlhf.data import (
    parse_openai_comparison,
    select_top_workers,
    balanced_subset,
    write_jsonl,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-users", type=int, default=5)
    parser.add_argument("--train-per-user", type=int, default=5373)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit('Install the LLM extras: pip install -e ".[llm]"') from exc

    ds = load_dataset("openai/summarize_from_feedback", "comparisons")
    train_records = [parse_openai_comparison(row) for row in ds["train"]]
    workers = select_top_workers(train_records, args.top_users)
    train = balanced_subset(train_records, workers, args.train_per_user, args.seed)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, str(output / "train.jsonl"))
    (output / "workers.txt").write_text("\n".join(workers) + "\n", encoding="utf-8")

    val_split = next((name for name in ("validation", "valid", "test") if name in ds), None)
    if val_split is not None:
        worker_set = set(workers)
        val = []
        for row in ds[val_split]:
            rec = parse_openai_comparison(row)
            if rec.worker in worker_set:
                val.append(rec)
        write_jsonl(val, str(output / "val.jsonl"))
        print(f"validation split: {val_split}; pairs for selected workers: {len(val)}")

    print(f"selected workers: {workers}")
    print(f"balanced training pairs: {len(train)}")


if __name__ == "__main__":
    main()

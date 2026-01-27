from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any
import json

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PreferenceRecord:
    worker: str
    prompt: str
    chosen: str
    rejected: str


def _worker(row: dict[str, Any]) -> str:
    for key in ("worker", "worker_id", "annotator", "annotator_id"):
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError("No worker/annotator id found in comparison row")


def _prompt(row: dict[str, Any]) -> str:
    info = row.get("info", {})
    if isinstance(info, dict):
        for key in ("post", "article", "text"):
            if info.get(key):
                return str(info[key])
    for key in ("prompt", "post", "article"):
        if row.get(key):
            return str(row[key])
    return ""


def parse_openai_comparison(row: dict[str, Any]) -> PreferenceRecord:
    """Normalize a summarize_from_feedback comparison row."""
    worker = _worker(row)
    prompt = _prompt(row)

    if "chosen" in row and "rejected" in row:
        chosen = row["chosen"]
        rejected = row["rejected"]
        if isinstance(chosen, dict):
            chosen = chosen.get("text", "")
        if isinstance(rejected, dict):
            rejected = rejected.get("text", "")
        return PreferenceRecord(worker, prompt, str(chosen), str(rejected))

    summaries = row.get("summaries")
    choice = row.get("choice")
    if isinstance(summaries, (list, tuple)) and len(summaries) == 2 and choice in (0, 1):
        texts = [s.get("text", "") if isinstance(s, dict) else str(s) for s in summaries]
        return PreferenceRecord(
            worker=worker,
            prompt=prompt,
            chosen=texts[int(choice)],
            rejected=texts[1 - int(choice)],
        )

    raise ValueError("Unsupported comparison schema")


def select_top_workers(records: list[PreferenceRecord], n: int) -> list[str]:
    counts = Counter(r.worker for r in records)
    return [worker for worker, _ in counts.most_common(n)]


def balanced_subset(
    records: list[PreferenceRecord],
    workers: list[str],
    per_worker: int,
    seed: int = 42,
) -> list[PreferenceRecord]:
    rng = np.random.default_rng(seed)
    out = []
    for worker in workers:
        idx = [i for i, r in enumerate(records) if r.worker == worker]
        if len(idx) < per_worker:
            raise ValueError(f"worker {worker} has only {len(idx)} records")
        chosen = rng.choice(idx, size=per_worker, replace=False)
        out.extend(records[int(i)] for i in chosen)
    rng.shuffle(out)
    return out


class PairEmbeddingDataset(Dataset):
    def __init__(self, path: str):
        data = np.load(path)
        self.chosen = torch.from_numpy(data["chosen"]).float()
        self.rejected = torch.from_numpy(data["rejected"]).float()
        self.user_id = torch.from_numpy(data["user_id"]).long()
        if not (len(self.chosen) == len(self.rejected) == len(self.user_id)):
            raise ValueError("embedding arrays have different lengths")

    def __len__(self):
        return len(self.user_id)

    def __getitem__(self, idx):
        return {
            "chosen": self.chosen[idx],
            "rejected": self.rejected[idx],
            "user_id": self.user_id[idx],
        }


def write_jsonl(records: list[PreferenceRecord], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")

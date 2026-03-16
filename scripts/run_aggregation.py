#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import numpy as np

from hetero_rlhf.aggregation import sweep_aggregation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rewards", required=True, help="NumPy matrix [..., users]")
    p.add_argument("--alphas", nargs="+", type=float, required=True)
    p.add_argument("--output")
    args = p.parse_args()

    rewards = np.load(args.rewards)
    results = sweep_aggregation(rewards, args.alphas)
    serializable = {
        str(alpha): np.asarray(values).tolist()
        for alpha, values in results.items()
    }
    text = json.dumps(serializable, indent=2)
    if args.output:
        open(args.output, "w", encoding="utf-8").write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()

"""Exploratory comparison of two conditions in a per-run table from analyze_minigrid.py.

Usage:
    python src/permutation_test.py results/tables/exp011_vs_exp012_runs.csv sparse view5_sparse

For each task and metric: means and medians of both conditions, the exact two-sided permutation test on
the difference of means (all splits of the pooled runs), and the probability of improvement
P(B > A) over all run pairs (ties count 1/2). For steps to threshold, a run that never reached the
threshold counts as the full budget (3M), so "higher" means slower.
"""
import argparse
import csv
import itertools

import numpy as np

METRICS = ("final_succ", "auc", "greedy_succ", "succ_len", "greedy_succ_len", "steps_50", "steps_80",
           "train_succ_500k")
CENSORED = {"steps_50": 3e6, "steps_80": 3e6}


def permutation_p(a, b):
    pooled = np.concatenate([a, b])
    observed = abs(b.mean() - a.mean())
    diffs = []
    for idx in itertools.combinations(range(len(pooled)), len(b)):
        mask = np.zeros(len(pooled), bool)
        mask[list(idx)] = True
        diffs.append(abs(pooled[mask].mean() - pooled[~mask].mean()))
    return float(np.mean(np.array(diffs) >= observed - 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("table")
    parser.add_argument("cond_a")
    parser.add_argument("cond_b")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.table)))
    for task in sorted({r["task"] for r in rows}, key=lambda t: ("Lava" in t, t)):
        print(f"{task}: {args.cond_b} vs {args.cond_a}")
        for key in METRICS:
            def values(cond):
                return np.array([float(r[key]) if r[key] != "" else CENSORED[key]
                                 for r in rows if r["task"] == task and r["condition"] == cond])
            a, b = values(args.cond_a), values(args.cond_b)
            poi = np.mean([(y > x) + 0.5 * (y == x) for x in a for y in b])
            print(f"  {key:16s} A {a.mean():10.3f} (median {np.median(a):10.3f})   "
                  f"B {b.mean():10.3f} (median {np.median(b):10.3f})   "
                  f"diff {b.mean() - a.mean():+10.3f}   p = {permutation_p(a, b):.3f}   P(B > A) = {poi:.2f}")


if __name__ == "__main__":
    main()

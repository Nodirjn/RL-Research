"""Exploratory hypothesis checks of EXP-020 (the table in documentation/stages/08_shaping_experiment.md).

Usage (from the project root):
    python src/hypothesis_checks.py --name exp020

Merges the per-run tables of analyze_minigrid.py, loop_metrics.py and behavior.py
(results/tables/<name>_runs.csv, _loops.csv, _crossing.csv) and reports, for each pre-registered check,
the difference of means with its 95% bootstrap interval and P(B > A), using the functions of
compare_conditions.py. No test: these checks are exploratory.
never_crossed: fraction of the timed-out episodes in which the agent never stood on the goal side of the
line (as in behavior.py; a run without timeouts has no value and is dropped). never_crossed_rate: the same
episodes as a fraction of all 100 final-evaluation episodes (defined for every run).
Writes results/tables/<name>_hypotheses.csv.
"""
import argparse
import csv
from pathlib import Path

from compare_conditions import compare

SIMPLE, LAVA = "SimpleCrossingS9N1", "LavaCrossingS9N1"
OUTCOME_SHAPING = ["shaping_success", "shaping_death", "shaping_timeout"]
# (hypothesis, tasks, condition A, condition B, metrics), B compared against A.
CHECKS = [
    ("H-dead-end", (SIMPLE, LAVA), "euclidean", "bfs",
     ["stochastic_never_crossed", "greedy_never_crossed", "stochastic_never_crossed_rate",
      "greedy_never_crossed_rate", "revisit_ratio"]),
    ("H-lava-a", (LAVA,), "sparse", "euclidean", ["train_death_500k"]),
    ("H-lava-b", (LAVA,), "euclidean", "euclidean_lava",
     ["stochastic_never_crossed", "greedy_never_crossed", "stochastic_never_crossed_rate",
      "greedy_never_crossed_rate"]),
    ("H-naive", (SIMPLE, LAVA), "euclidean", "naive",
     ["greedy_loop_rate", "revisit_ratio"] + OUTCOME_SHAPING),
]


def merged_rows(name):
    tables = Path("results/tables")
    runs = {r["run"]: r for r in csv.DictReader(open(tables / f"{name}_runs.csv"))}
    for extra in ("loops", "crossing"):
        for r in csv.DictReader(open(tables / f"{name}_{extra}.csv")):
            runs[r["run"]].update({k: v for k, v in r.items() if k not in runs[r["run"]]})
    for r in runs.values():
        for mode in ("greedy", "stochastic"):
            timeouts, never = int(r[f"{mode}_timeouts"]), int(r[f"{mode}_timeouts_never_crossed"])
            r[f"{mode}_never_crossed"] = never / timeouts if timeouts else ""
            r[f"{mode}_never_crossed_rate"] = never / 100
    return list(runs.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    rows = merged_rows(args.name)
    results = []
    for hypothesis, tasks, cond_a, cond_b, metrics in CHECKS:
        for task in tasks:
            for metric in metrics:
                if not any(r["task"] == task and r["condition"] in (cond_a, cond_b) and r[metric] != ""
                           for r in rows):
                    continue  # e.g. no deaths on SimpleCrossing
                try:
                    results.append({"hypothesis": hypothesis, **compare(rows, task, cond_a, cond_b, metric)})
                except SystemExit as e:
                    print(f"{hypothesis} {task} {metric}: skipped ({e})")
    for r in results:
        print(f"{r['hypothesis']:11s} {r['task']:19s} {r['B']:>14s} vs {r['A']:<10s} {r['metric']:30s} "
              f"{r['mean_a']:7.3f} -> {r['mean_b']:7.3f}  diff {r['diff']:+.3f} [{r['ci_low']:+.3f}, "
              f"{r['ci_high']:+.3f}]  P(B > A) {r['poi']:.2f}  n {r['n_a']}/{r['n_b']}")
    out = Path("results/tables") / f"{args.name}_hypotheses.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\ntable: {out}")


if __name__ == "__main__":
    main()

"""RQ4 of EXP-021: does shaping help more with the 5×5 view? (documentation/stages/10_view_shaping_experiment.md)

Usage (from the project root, after analyze_minigrid.py, loop_metrics.py and behavior.py have written the
tables of both views):
    python src/view_interaction.py --view7 exp020 --view5 exp021

The 7×7 groups (sparse, euclidean, bfs) come from the <view7> tables, the 5×5 groups (view5_sparse,
view5_euclidean, view5_bfs) from the <view5> tables. For each task and potential S, the interaction contrast

    I = (S − sparse with the 5×5 view) − (S − sparse with the 7×7 view)

with a percentile bootstrap interval (compare_conditions.interaction). Three kinds of rows:
    confirmatory   stochastic_succ_steps_after_cross: steps from the first step on the goal side of the line to
                   the goal, mean over the successful stochastic episodes. The four contrasts (2 tasks ×
                   2 potentials) get 98.75% intervals (Bonferroni over four). I < 0: shaping shortens the walk
                   to the goal more with the 5×5 view. A claim needs the whole interval below 0.
    estimate       the primary metrics (AUC, final success, steps to 50% and 80%), 95% intervals, no claim.
    exploratory    other behavior metrics, 95% intervals.
after_cross_timeout_rate: stochastic timeouts in which the agent had crossed the line, as a fraction of all
100 final-evaluation episodes.
Writes results/tables/<view5>_interactions.csv.
"""
import argparse
import csv
from pathlib import Path

from compare_conditions import PRIMARY, interaction
from hypothesis_checks import LAVA, SIMPLE, merged_rows

POTENTIALS = ("euclidean", "bfs")
CONFIRMATORY = "stochastic_succ_steps_after_cross"
CONFIRMATORY_LEVEL = 1 - 0.05 / 4
EXPLORATORY = ("greedy_succ_steps_after_cross", "stochastic_succ_steps_to_cross", "after_cross_timeout_rate",
               "revisit_ratio", "greedy_loop_rate")
PLAN = ([("confirmatory", CONFIRMATORY, CONFIRMATORY_LEVEL)] + [("estimate", m, 0.95) for m in PRIMARY]
        + [("exploratory", m, 0.95) for m in EXPLORATORY])


def view_rows(view7, view5):
    rows = merged_rows(view7) + merged_rows(view5)
    for r in rows:
        crossed = int(r["stochastic_timeouts"]) - int(r["stochastic_timeouts_never_crossed"])
        r["after_cross_timeout_rate"] = crossed / 100
    return rows


def run_plan(rows):
    results = []
    for kind, metric, level in PLAN:
        for task in (SIMPLE, LAVA):
            for s in POTENTIALS:
                try:
                    results.append({"kind": kind, **interaction(rows, task, "sparse", s, "view5_sparse",
                                                                 f"view5_{s}", metric, level)})
                except SystemExit as e:
                    if kind == "confirmatory":
                        raise
                    print(f"{kind} {task} {s} {metric}: skipped ({e})")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view7", required=True)
    parser.add_argument("--view5", required=True)
    args = parser.parse_args()
    results = run_plan(view_rows(args.view7, args.view5))
    for r in results:
        print(f"{r['kind']:12s} {r['task']:19s} {r['B5'][6:]:9s} {r['metric']:34s} 7×7 {r['effect7']:+9.3f}  "
              f"5×5 {r['effect5']:+9.3f}  I {r['interaction']:+9.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] "
              f"({r['level']:.2%})  n {r['n_a7']}/{r['n_b7']}/{r['n_a5']}/{r['n_b5']}")
    out = Path("results/tables") / f"{args.view5}_interactions.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\ntable: {out}")


if __name__ == "__main__":
    main()

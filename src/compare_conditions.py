"""Pre-registered comparisons between conditions (EXP-020 onwards), from a per-run table of analyze_minigrid.py.

Usage:
    python src/compare_conditions.py results/tables/exp020_runs.csv --family exp020
    python src/compare_conditions.py results/tables/exp020_loops.csv --pairs euclidean:bfs --metrics revisit_ratio

--family runs the confirmatory comparisons of the pre-registration (FAMILIES): for each comparison "B against
A", the exact two-sided permutation p of the difference of mean AUC, Holm-corrected over the whole family;
the other primary metrics (PRIMARY) are reported for the same pairs without a test. --pairs reports any
other comparison (exploratory) without a test.

For every comparison and metric: the means, the difference B - A with a 95% percentile bootstrap interval
(10,000 resamples of the runs within each condition, fixed seed), and the probability of improvement
P(B > A) over all run pairs (ties count 1/2) with its bootstrap interval. For steps to a threshold, a run
that never reached it counts as the budget (3M), so "higher" means slower. Other empty values are dropped.
Writes <table stem>_comparisons.csv next to the table.

interaction() gives the view-size interaction contrasts of EXP-021 (used by view_interaction.py).
"""
import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from permutation_test import CENSORED

CONFIRMATORY_METRIC = "auc"
PRIMARY = ("auc", "final_succ", "steps_50", "steps_80")
N_BOOTSTRAP = 10_000
# (task, condition A, condition B): B is compared against A (documentation/stages/08_shaping_experiment.md).
FAMILIES = {
    "exp020": [
        (task, a, b)
        for task in ("SimpleCrossingS9N1", "LavaCrossingS9N1")
        for a, b in (("sparse", "euclidean"), ("sparse", "bfs"), ("euclidean", "bfs"),
                     ("euclidean", "euclidean_lava"), ("euclidean", "naive"))
        if not (b == "euclidean_lava" and task == "SimpleCrossingS9N1")  # no lava on SimpleCrossing
    ],
    # The same potentials with the 5×5 view (documentation/stages/10_view_shaping_experiment.md).
    "exp021": [
        (task, a, b)
        for task in ("SimpleCrossingS9N1", "LavaCrossingS9N1")
        for a, b in (("view5_sparse", "view5_euclidean"), ("view5_sparse", "view5_bfs"),
                     ("view5_euclidean", "view5_bfs"))
    ],
}


def values(rows, task, condition, metric):
    out = []
    for r in rows:
        if r["task"] == task and r["condition"] == condition:
            if r[metric] != "":
                out.append(float(r[metric]))
            elif metric in CENSORED:
                out.append(CENSORED[metric])
    return np.array(out)


def permutation_p(a, b):
    """Exact two-sided permutation p of the difference of means over all splits of the pooled runs.

    The test of permutation_test.permutation_p, vectorized (10 against 10 runs have 184,756 splits).
    """
    pooled = np.concatenate([a, b])
    idx = np.array(list(itertools.combinations(range(len(pooled)), len(b))))
    sum_b = pooled[idx].sum(axis=1)
    diffs = np.abs(sum_b / len(b) - (pooled.sum() - sum_b) / len(a))
    return float(np.mean(diffs >= abs(b.mean() - a.mean()) - 1e-12))


def probability_of_improvement(a, b):
    """P(B > A) over all pairs of runs; ties count 1/2. Works on the last two axes for batches."""
    a, b = a[..., :, None], b[..., None, :]
    return ((b > a) + 0.5 * (b == a)).mean(axis=(-2, -1))


def bootstrap(a, b, seed=0):
    """95% percentile intervals of the difference of means and of P(B > A), resampling runs within groups."""
    rng = np.random.default_rng(seed)
    ra = a[rng.integers(0, len(a), (N_BOOTSTRAP, len(a)))]
    rb = b[rng.integers(0, len(b), (N_BOOTSTRAP, len(b)))]
    diff = rb.mean(axis=1) - ra.mean(axis=1)
    poi = probability_of_improvement(ra, rb)
    return tuple(np.percentile(diff, [2.5, 97.5])), tuple(np.percentile(poi, [2.5, 97.5]))


def interaction(rows, task, a7, b7, a5, b5, metric, level=0.95, seed=0):
    """(B − A with the 5×5 view) − (B − A with the 7×7 view), with a percentile bootstrap interval.

    The runs are resampled within each of the four groups; level is the coverage of the interval.
    """
    names = (a7, b7, a5, b5)
    groups = [values(rows, task, c, metric) for c in names]
    for c, g in zip(names, groups):
        if len(g) == 0:
            raise SystemExit(f"no runs for {task}: {c} ({metric})")
    rng = np.random.default_rng(seed)
    m = [g[rng.integers(0, len(g), (N_BOOTSTRAP, len(g)))].mean(axis=1) for g in groups]
    tail = 100 * (1 - level) / 2
    lo, hi = np.percentile((m[3] - m[2]) - (m[1] - m[0]), [tail, 100 - tail])
    effect7, effect5 = groups[1].mean() - groups[0].mean(), groups[3].mean() - groups[2].mean()
    return {"task": task, "A7": a7, "B7": b7, "A5": a5, "B5": b5, "metric": metric,
            **{f"n_{k}": len(g) for k, g in zip(("a7", "b7", "a5", "b5"), groups)},
            **{f"mean_{k}": g.mean() for k, g in zip(("a7", "b7", "a5", "b5"), groups)},
            "effect7": effect7, "effect5": effect5, "interaction": effect5 - effect7,
            "level": level, "ci_low": lo, "ci_high": hi}


def holm(p_values):
    """Holm-corrected p-values, in the original order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def compare(rows, task, cond_a, cond_b, metric, test=False):
    a, b = values(rows, task, cond_a, metric), values(rows, task, cond_b, metric)
    if len(a) == 0 or len(b) == 0:
        raise SystemExit(f"no runs for {task}: {cond_a if len(a) == 0 else cond_b} ({metric})")
    (lo, hi), (poi_lo, poi_hi) = bootstrap(a, b)
    return {"task": task, "A": cond_a, "B": cond_b, "metric": metric, "n_a": len(a), "n_b": len(b),
            "mean_a": a.mean(), "mean_b": b.mean(), "diff": b.mean() - a.mean(), "ci_low": lo, "ci_high": hi,
            "poi": float(probability_of_improvement(a, b)), "poi_low": poi_lo, "poi_high": poi_hi,
            "p": permutation_p(a, b) if test else None, "p_holm": None}


def run_family(rows, family):
    results = [compare(rows, task, a, b, metric, test=metric == CONFIRMATORY_METRIC)
               for task, a, b in FAMILIES[family] for metric in PRIMARY]
    tested = [r for r in results if r["p"] is not None]
    for r, adjusted in zip(tested, holm([r["p"] for r in tested])):
        r["p_holm"] = adjusted
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("table")
    parser.add_argument("--family", choices=sorted(FAMILIES))
    parser.add_argument("--pairs", nargs="+", default=[], help="exploratory pairs A:B")
    parser.add_argument("--metrics", nargs="+", default=list(PRIMARY), help="metrics for --pairs")
    args = parser.parse_args()
    if not args.family and not args.pairs:
        parser.error("give --family or --pairs")

    rows = list(csv.DictReader(open(args.table)))
    results = run_family(rows, args.family) if args.family else []
    tasks = sorted({r["task"] for r in rows}, key=lambda t: ("Lava" in t, t))
    for pair in args.pairs:
        cond_a, cond_b = pair.split(":")
        results += [compare(rows, task, cond_a, cond_b, metric) for task in tasks for metric in args.metrics
                    if len(values(rows, task, cond_a, metric)) and len(values(rows, task, cond_b, metric))]

    for r in results:
        test = f"   p = {r['p']:.4f}, Holm {r['p_holm']:.4f}" if r["p"] is not None else ""
        print(f"{r['task']:20s} {r['B']:>15s} vs {r['A']:<10s} {r['metric']:12s} "
              f"diff {r['diff']:+10.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]   "
              f"P(B > A) {r['poi']:.2f} [{r['poi_low']:.2f}, {r['poi_high']:.2f}]   n {r['n_a']}/{r['n_b']}{test}")
    out = Path(args.table).with_name(Path(args.table).stem + "_comparisons.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\ncomparisons: {out}")


if __name__ == "__main__":
    main()

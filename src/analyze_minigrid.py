"""Pre-registered metrics and success curves for MiniGrid experiments (EXP-011 onwards).

Usage:
    python src/analyze_minigrid.py experiments/EXP-011 --name exp011
    python src/analyze_minigrid.py experiments/EXP-011 experiments/EXP-012 --name exp011_vs_exp012

Per run (see the EXP-011 pre-registration in RESEARCH_LOG.md):
    final_succ     stochastic success on 100 held-out episodes (final_eval.json)
    auc            mean periodic-eval success over all evaluations from 0 to the budget (0-1)
    steps_50/80    first scheduled evaluation step with success >= 0.5 / 0.8 (empty if never reached)
    greedy_succ    deterministic success on the same 100 episodes (final_eval_greedy.json)
    death, timeout termination rates of the final evaluations; succ_len = mean length of successful episodes
    train_*_500k   descriptive, not pre-registered: outcome rates of the training episodes that finished
                   in the first 500k steps (from train_monitor.csv: return > 0 = success, return 0 before the
                   time limit = death, return 0 at the time limit = timeout)
Per condition: mean ± SD (ddof=1) across seeds; steps_50/80 over the runs that reached the threshold;
the same training-episode outcome rates per step window (TRAIN_BINS), each the mean of the per-run rates.
Writes results/tables/<name>_runs.csv and results/figures/<name>_success.png.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import budget_label, load_config, make_env

EARLY_STEPS = 500_000
TRAIN_BINS = (0, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000, 3_000_000)
# Fixed color per condition (validated categorical slots), so a condition keeps its color in every figure.
COLORS = {"sparse": "#2a78d6", "view5_sparse": "#eb6834"}
LABELS = {"sparse": "7×7 view, sparse reward", "view5_sparse": "5×5 view, sparse reward"}
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def first_step_at(evals, threshold):
    for r in evals:
        if float(r["success_rate"]) >= threshold:
            return int(r["scheduled_step"])
    return None


def training_outcomes(run_dir, max_steps):
    """(end step, outcome) of every training episode, in completion order.

    The running sum of episode lengths approximates the global step at which each episode ended
    (off by at most the steps of the episodes still in progress in the other envs). Runs without an
    `outcome` column (no shaping) are classified by return and length, which is exact for the original reward.
    """
    with open(run_dir / "train_monitor.csv") as f:
        next(f)  # VecMonitor's JSON header line
        rows = list(csv.DictReader(f))
    steps, outcomes = 0, []
    for r in rows:
        ret, length = float(r["r"]), int(r["l"])
        steps += length
        if "outcome" in r:  # runs with a `shaping` section: `r` is the shaped return, use the env's outcome
            outcomes.append((steps, r["outcome"]))
        else:
            outcomes.append((steps, "success" if ret > 0 else "timeout" if length >= max_steps else "death"))
    return outcomes


def outcome_rates(outcomes, start, end):
    """Success / death / timeout rates of the training episodes that ended in (start, end]."""
    window = [o for step, o in outcomes if start < step <= end]
    n = len(window)
    return {k: window.count(k) / n if n else None for k in ("success", "death", "timeout")}, n


def analyze_run(run_dir):
    cfg = load_config(run_dir / "config.yaml")
    evals = list(csv.DictReader(open(run_dir / "eval.csv")))
    final = json.load(open(run_dir / "final_eval.json"))
    greedy = json.load(open(run_dir / "final_eval_greedy.json"))
    env = make_env(cfg)  # imports minigrid first, which registers the envs
    max_steps = env.unwrapped.max_steps
    env.close()
    outcomes = training_outcomes(run_dir, max_steps)
    early, n_early = outcome_rates(outcomes, 0, EARLY_STEPS)

    def succ_len(result):
        lengths = [e["length"] for e in result["episodes"] if e["success"]]
        return float(np.mean(lengths)) if lengths else None

    task = cfg["env_id"].replace("MiniGrid-", "").replace("-v0", "")
    return {
        "run": run_dir.name, "task": task, "condition": cfg["condition"].replace(f"{task}_", ""),
        "budget": budget_label(cfg["total_timesteps"]), "seed": cfg["seed"],
        "final_succ": final["success_rate"], "auc": float(np.mean([float(r["success_rate"]) for r in evals])),
        "steps_50": first_step_at(evals, 0.5), "steps_80": first_step_at(evals, 0.8),
        "greedy_succ": greedy["success_rate"],
        "death": final["death_rate"], "timeout": final["timeout_rate"],
        "greedy_death": greedy["death_rate"], "greedy_timeout": greedy["timeout_rate"],
        "succ_len": succ_len(final), "greedy_succ_len": succ_len(greedy),
        "train_succ_500k": early["success"], "train_death_500k": early["death"],
        "train_timeout_500k": early["timeout"], "train_episodes_500k": n_early,
        "_curve": [(int(r["scheduled_step"]), float(r["success_rate"])) for r in evals],
        "_bins": [outcome_rates(outcomes, a, b)[0] for a, b in zip(TRAIN_BINS, TRAIN_BINS[1:])],
    }


def fmt(values, digits=3):
    values = [v for v in values if v is not None]
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def print_summary(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[(r["task"], r["condition"], r["budget"])].append(r)
    for (task, cond, budget), rs in groups.items():
        print(f"\n{task} | {cond} | {budget} | {len(rs)} seeds (mean ± SD across seeds)")
        for key in ("final_succ", "auc", "greedy_succ", "death", "timeout", "greedy_death", "greedy_timeout",
                    "train_succ_500k", "train_death_500k", "train_timeout_500k"):
            print(f"  {key:20s} {fmt([r[key] for r in rs])}")
        for key in ("succ_len", "greedy_succ_len"):
            print(f"  {key:20s} {fmt([r[key] for r in rs], 1)}")
        for key in ("steps_50", "steps_80"):
            reached = [r[key] / 1e6 for r in rs if r[key] is not None]
            print(f"  {key:20s} {fmt(reached, 2)} M  (reached by {len(reached)}/{len(rs)} seeds)")
        print("  training episodes, success / death / timeout (mean of per-run rates):")
        for i, (a, b) in enumerate(zip(TRAIN_BINS, TRAIN_BINS[1:])):
            rates = [np.mean([r["_bins"][i][k] for r in rs if r["_bins"][i][k] is not None])
                     for k in ("success", "death", "timeout")]
            window = f"{budget_label(a) if a else '0'}–{budget_label(b)}"
            print(f"    {window:10s} " + " / ".join(f"{x:.2f}" for x in rates))


def plot(runs, out):
    tasks = sorted({r["task"] for r in runs}, key=lambda t: ("Lava" in t, t))
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 3.6), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, tasks):
        ax.set_facecolor(SURFACE)
        conditions = sorted({r["condition"] for r in runs if r["task"] == task}, key=list(COLORS).index)
        for cond in conditions:
            rs = [r for r in runs if r["task"] == task and r["condition"] == cond]
            color = COLORS[cond]
            for r in rs:
                x, y = zip(*r["_curve"])
                ax.plot(np.array(x) / 1e6, y, color=color, lw=0.8, alpha=0.35)
            steps = sorted({s for r in rs for s, _ in r["_curve"]})
            mean = [np.mean([dict(r["_curve"])[s] for r in rs if s in dict(r["_curve"])]) for s in steps]
            ax.plot(np.array(steps) / 1e6, mean, color=color, lw=2, label=f"{LABELS.get(cond, cond)}: mean of {len(rs)} seeds")
        ax.set_title(task, color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("environment steps (millions)", color=TEXT_2)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=TEXT_2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0].set_ylabel("eval success rate\n(stochastic policy, 20 episodes)", color=TEXT_2)
    # One legend under the panels, so it never covers the curves.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=TEXT_2, loc="lower center",
               ncol=len(handles), title="thin lines: single seeds", title_fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"\nfigure: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dirs", nargs="+")
    parser.add_argument("--name", required=True, help="file stem for the table and the figure")
    args = parser.parse_args()

    runs = [analyze_run(d) for e in args.exp_dirs for d in sorted(Path(e).iterdir())
            if (d / "final_eval_greedy.json").exists()]
    if not runs:
        raise SystemExit("no finished runs")
    table = Path("results/tables") / f"{args.name}_runs.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    fields = [k for k in runs[0] if not k.startswith("_")]
    with open(table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    print(f"per-run table: {table}")
    print_summary(runs)
    plot(runs, Path("results/figures") / f"{args.name}_success.png")


if __name__ == "__main__":
    main()

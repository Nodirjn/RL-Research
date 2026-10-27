"""Plot evaluation learning curves (mean across seeds, min–max band) with published reference points.

Usage:
    python src/plot_curves.py experiments/EXP-001 --groups C1_hist_3M C2_cleanrl_3M C3_zoo_3M \
        --out results/figures/exp001_curves.png

A group is `<condition>_<budget>` as written by train.py; every run of that group is averaged.
Left panel: deterministic eval return (10 episodes per point). Right panel: mean episode length.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load_config

# Fixed color per condition (validated categorical slots 1-3), so a condition keeps its color in every figure.
COLORS = {"C1_hist": "#2a78d6", "C2_cleanrl": "#eb6834", "C3_zoo": "#1baf7a"}
SURFACE, TEXT, TEXT_2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
# Published PPO results on Humanoid (see RESEARCH_LOG.md): (label, marker, step, mean, sd or None).
# One marker per source, so the gray markers stay distinguishable in the legend.
REFERENCES = [
    ("Gao et al. v5 (10 seeds)", "D", 1_000_000, 671.1, 128.3),
    ("Gao et al. v5 (10 seeds)", "D", 3_000_000, 941.5, 288.5),
    ("CleanRL v4 (1 seed)", "s", 1_000_000, 378.9, None),
    ("SLM-Lab v5 (16 envs, moving avg.)", "^", 10_000_000, 2661.3, None),
]


def load_group(exp_dir, group):
    curves = []
    for run_dir in sorted(Path(exp_dir).iterdir()):
        if not (run_dir / "eval.csv").exists() or not (run_dir / "config.yaml").exists():
            continue
        cfg = load_config(run_dir / "config.yaml")
        if f"{cfg['condition']}_{cfg['total_timesteps'] // 1_000_000}M" != group:
            continue
        rows = list(csv.DictReader(open(run_dir / "eval.csv")))
        curves.append({int(r["scheduled_step"]): (float(r["mean_return"]), float(r["mean_length"])) for r in rows})
    if not curves:
        raise SystemExit(f"no runs found for group {group}")
    steps = sorted(set.intersection(*(set(c) for c in curves)))  # points every seed has reached
    values = np.array([[c[s] for s in steps] for c in curves])  # (seeds, steps, 2)
    return np.array(steps), values, len(curves)


def place_end_labels(ax, labels, min_gap=0.045):
    """Label each curve at its last point, pushing labels apart so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = min_gap * (hi - lo)
    labels = sorted(labels)
    ys = [y for y, _, _ in labels]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (_, x, text), y in zip(labels, ys):
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=TEXT_2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
                         "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT})
    fig, (ax_ret, ax_len) = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    xmax = 0
    end_labels = {ax_ret: [], ax_len: []}
    for group in args.groups:
        steps, values, n_seeds = load_group(args.exp_dir, group)
        color = COLORS[group.rsplit("_", 1)[0]]
        label = f"{group} (n={n_seeds})"
        xmax = max(xmax, steps[-1])
        for ax, k in ((ax_ret, 0), (ax_len, 1)):
            v = values[:, :, k]
            ax.fill_between(steps, v.min(0), v.max(0), color=color, alpha=0.15, linewidth=0)
            ax.plot(steps, v.mean(0), color=color, linewidth=2, label=label)
            end_labels[ax].append((v.mean(0)[-1], steps[-1], group.split("_")[0]))

    # Reference points are identified in the legend; text next to them would collide with the curves.
    for name, marker, step, mean, sd in REFERENCES:
        if step <= xmax:
            label = name if name not in ax_ret.get_legend_handles_labels()[1] else None
            ax_ret.errorbar(step, mean, yerr=sd, fmt=marker, markersize=7, color=NEUTRAL, ecolor=NEUTRAL,
                            elinewidth=1.5, capsize=4, zorder=5, label=label)
    ax_len.axhline(1000, color=NEUTRAL, linestyle="--", linewidth=1)
    ax_len.annotate("time limit (1000 steps)", (0, 1000), xytext=(4, -12), textcoords="offset points",
                    fontsize=8, color=TEXT_2)

    for ax, title, ylabel in ((ax_ret, "Deterministic eval return", "mean return (10 episodes)"),
                              (ax_len, "Episode length", "mean episode length (steps)")):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", fontsize=11, color=TEXT)
        ax.set_xlabel("environment steps")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, xmax * 1.08)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M"))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        place_end_labels(ax, end_labels[ax])
    ax_ret.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.01, 0.01, "Lines: mean across seeds; bands: min–max across seeds. Humanoid-v5, original reward. "
             "Gray markers: published PPO results (mean ± SD where reported).", fontsize=8, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

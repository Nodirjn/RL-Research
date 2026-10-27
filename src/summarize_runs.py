"""Summarize finished runs of an experiment into per-run and per-condition tables.

Usage:
    python src/summarize_runs.py experiments/EXP-001 --milestones 1000000 3000000

For each milestone M, `last5@M` is the mean return of the 5 periodic evaluations scheduled at
M-4*freq ... M, matching the "average of the final five evaluations" metric of Gao et al.
(arXiv 2603.10199). Next to it is the task's headline rate over the same evaluations: the fall
rate (MuJoCo locomotion) or the success rate (MiniGrid). `final50` is the held-out evaluation of
the final policy (50 episodes in the Humanoid experiments; `final` columns hold the same for MiniGrid).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import budget_label, load_config


def last5(evals, milestone, freq, rate_key):
    wanted = {milestone - k * freq for k in range(5)}
    rows = [r for r in evals if int(r["scheduled_step"]) in wanted]
    if len(rows) < 5:
        return None
    return float(np.mean([float(r["mean_return"]) for r in rows])), float(np.mean([float(r[rate_key]) for r in rows]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("--milestones", type=int, nargs="+", required=True)
    parser.add_argument("--out", help="CSV path for the per-run table")
    args = parser.parse_args()

    runs = []
    for run_dir in sorted(p for p in Path(args.exp_dir).iterdir() if (p / "final_eval.json").exists()):
        cfg = load_config(run_dir / "config.yaml")
        evals = list(csv.DictReader(open(run_dir / "eval.csv")))
        final = json.load(open(run_dir / "final_eval.json"))
        meta = json.load(open(run_dir / "meta.json"))
        locomotion = "fall_rate" in final
        rate_key, rate_name = ("fall_rate", "fall") if locomotion else ("success_rate", "succ")
        row = {"run": run_dir.name, "condition": f"{cfg['condition']}_{budget_label(cfg['total_timesteps'])}",
               "seed": cfg["seed"], "timesteps": final["timesteps"], "steps_per_s": meta["steps_per_second"],
               "train_h": round(meta["train_seconds"] / 3600, 2)}
        for m in args.milestones:
            value = last5(evals, m, cfg["eval"]["freq"], rate_key)
            row[f"last5@{budget_label(m)}"] = round(value[0], 1 if locomotion else 3) if value else ""
            row[f"{rate_name}@{budget_label(m)}"] = round(value[1], 2) if value else ""
        if locomotion:
            row.update(final50=round(final["mean_return"], 1), final50_sd=round(final["std_return"], 1),
                       final50_len=round(final["mean_length"], 1), final50_fall=round(final["fall_rate"], 2),
                       final50_xvel=round(final["mean_x_velocity"], 2))
        else:
            row.update(final_return=round(final["mean_return"], 3), final_success=round(final["success_rate"], 3),
                       final_death=round(final["death_rate"], 3), final_len=round(final["mean_length"], 1))
        runs.append(row)

    if not runs:
        raise SystemExit(f"no finished runs in {args.exp_dir}")
    fields = list(runs[0])
    print("Per run:")
    print("  " + " | ".join(fields))
    for r in runs:
        print("  " + " | ".join(str(r[f]) for f in fields))

    print("\nPer condition (mean ± SD across seeds):")
    groups = defaultdict(list)
    for r in runs:
        groups[r["condition"]].append(r)
    metrics = [f for f in fields if f.startswith(("last5@", "succ@", "final"))] + ["steps_per_s"]
    for cond, rs in groups.items():
        parts = []
        for f in metrics:
            vals = [r[f] for r in rs if r[f] != ""]
            if vals:
                digits = 1 if np.mean(np.abs(vals)) >= 10 else 3
                parts.append(f"{f}={np.mean(vals):.{digits}f}±{np.std(vals, ddof=1) if len(vals) > 1 else 0:.{digits}f}")
        print(f"  {cond} (n={len(rs)}): " + "  ".join(parts))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(runs)


if __name__ == "__main__":
    main()

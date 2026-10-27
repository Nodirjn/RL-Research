"""Loop metrics of final MiniGrid policies (the EXP-011 addendum, also used for EXP-012).

Usage:
    python src/loop_metrics.py experiments/EXP-011 experiments/EXP-012 --name exp011_exp012

For every finished run, the final model is rolled out again on the final-evaluation seeds
(20000-20099 in the configs):
    greedy       deterministic policy. The outcome rates must equal final_eval_greedy.json exactly
                 (same seeds, same policy); a mismatch stops the script.
    stochastic   sampled actions, torch seeded with TORCH_SEED, so the numbers are reproducible. The
                 sampled episodes differ from those in final_eval.json, so the primary success rates
                 stay those of final_eval.json; this re-run is only for the loop metrics.
Metrics per run (definitions in common.run_episodes):
    greedy_loop_rate          fraction of greedy episodes that revisit an (x, y, dir) state
    greedy_timeouts_looped    fraction of greedy timeouts that revisited a state (loops explain them)
    revisit_ratio             mean per-episode fraction of steps ending in a visited state (stochastic)
    revisit_ratio_success / _timeout   the same, over successful / timed-out stochastic episodes
Writes results/tables/<name>_loops.csv and the episodes to results/loop_metrics/<run>_<mode>.json.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from common import load_policy, run_episodes, summarize

TORCH_SEED = 0


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def analyze_run(run_dir, out_dir):
    model, normalize_obs, cfg = load_policy(run_dir)
    n, offset = cfg["eval"]["final_episodes"], cfg["eval"]["final_seed_offset"]

    greedy = run_episodes(model, cfg, n, offset, normalize_obs, deterministic=True)
    reference = json.load(open(run_dir / "final_eval_greedy.json"))
    for key in ("success", "died", "time_limit"):
        new = [e[key] for e in greedy]
        old = [e[key] for e in reference["episodes"]]
        if new != old:
            raise SystemExit(f"{run_dir.name}: greedy re-run differs from final_eval_greedy.json in '{key}'")

    torch.manual_seed(TORCH_SEED)
    stochastic = run_episodes(model, cfg, n, offset, normalize_obs, deterministic=False)

    for mode, episodes in (("greedy", greedy), ("stochastic", stochastic)):
        with open(out_dir / f"{run_dir.name}_{mode}.json", "w") as f:
            json.dump({"run": run_dir.name, "mode": mode, "seed_offset": offset, "torch_seed": TORCH_SEED,
                       **summarize(episodes), "episodes": episodes}, f, indent=1)

    g, s = summarize(greedy), summarize(stochastic)
    greedy_timeouts = [e for e in greedy if e["time_limit"]]
    task = cfg["env_id"].replace("MiniGrid-", "").replace("-v0", "")
    return {
        "run": run_dir.name, "task": task, "condition": cfg["condition"].replace(f"{task}_", ""),
        "seed": cfg["seed"],
        "greedy_succ": g["success_rate"], "greedy_timeout": g["timeout_rate"],
        "greedy_loop_rate": g["loop_rate"],
        "greedy_timeouts_looped": mean_or_none([e["revisited"] for e in greedy_timeouts]),
        "greedy_looped_but_succeeded": sum(e["revisited"] and e["success"] for e in greedy),
        "stoch_succ_rerun": s["success_rate"], "stoch_timeout_rerun": s["timeout_rate"],
        "revisit_ratio": s["revisit_ratio"],
        "revisit_ratio_success": mean_or_none([e["revisit_ratio"] for e in stochastic if e["success"]]),
        "revisit_ratio_timeout": mean_or_none([e["revisit_ratio"] for e in stochastic if e["time_limit"]]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dirs", nargs="+")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    out_dir = Path("results/loop_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for exp in args.exp_dirs:
        for run_dir in sorted(Path(exp).iterdir()):
            if (run_dir / "final_eval_greedy.json").exists():
                rows.append(analyze_run(run_dir, out_dir))
                r = rows[-1]
                print(f"{r['run']:42s} greedy loop {r['greedy_loop_rate']:.2f}  revisit {r['revisit_ratio']:.3f}",
                      flush=True)
    table = Path("results/tables") / f"{args.name}_loops.csv"
    with open(table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"table: {table}")

    groups = {}
    for r in rows:
        groups.setdefault((r["task"], r["condition"]), []).append(r)
    for (task, cond), rs in groups.items():
        print(f"\n{task} | {cond} | {len(rs)} seeds (mean ± SD; per-seed values in brackets)")
        for key in ("greedy_loop_rate", "greedy_timeouts_looped", "greedy_looped_but_succeeded", "revisit_ratio",
                    "revisit_ratio_success", "revisit_ratio_timeout", "stoch_succ_rerun"):
            values = [r[key] for r in rs if r[key] is not None]
            if not values:
                print(f"  {key:28s} —  (no such episodes)")
                continue
            spread = f" ± {np.std(values, ddof=1):.3f}" if len(values) > 1 else ""
            print(f"  {key:28s} {np.mean(values):.3f}{spread}  [{', '.join(f'{v:.2f}' for v in values)}]"
                  f"{'' if len(values) == len(rs) else f'  ({len(values)}/{len(rs)} seeds have such episodes)'}")


if __name__ == "__main__":
    main()

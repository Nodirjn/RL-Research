"""Where do the final MiniGrid policies fail? Line-crossing analysis, path figure and GIFs (workflow step 3).

Usage:
    python src/behavior.py experiments/EXP-011 experiments/EXP-012 --name exp011_exp012

Every final model is rolled out exactly as in loop_metrics.py (greedy, then stochastic with torch seed 0,
on the final-evaluation seeds), now with trajectories recorded. The outcomes and revisit ratios must equal
the records in results/loop_metrics/; a mismatch stops the script.

The S9N1 maps have one obstacle line (lava or wall) with one gap, and the goal is always behind it.
Per episode:
    crossed          the agent stood on the goal side of the line at some step
    steps_to_cross   steps until it first did
Per run, over the timed-out episodes: the fraction that never crossed.
Writes results/tables/<name>_crossing.csv, results/figures/<name>_paths.png (example paths) and GIFs
of the same examples to results/videos/ (the first GIF_STEPS steps; a loop only repeats after that).
"""
import argparse
import csv
import json
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import load_policy, make_env, run_episodes

TORCH_SEED = 0  # as in loop_metrics.py
GIF_STEPS = 120
SURFACE, TEXT, TEXT_2 = "#fcfcfb", "#0b0b0b", "#52514e"
CELL_COLORS = {"wall": "#5b5a56", "lava": "#e0703a", "goal": "#3a9a5b"}


def obstacle_line(env):
    """(axis, position, gap) of the single obstacle line: axis 'x' = a vertical line at x = position."""
    base = env.unwrapped
    cells = [(x, y) for x in range(1, base.width - 1) for y in range(1, base.height - 1)
             if base.grid.get(x, y) is not None and base.grid.get(x, y).type in ("lava", "wall")]
    xs, ys = {x for x, _ in cells}, {y for _, y in cells}
    axis, pos = ("x", xs.pop()) if len(xs) == 1 else ("y", ys.pop())
    span = range(1, (base.height if axis == "x" else base.width) - 1)
    gap = next((pos, k) if axis == "x" else (k, pos) for k in span
               if base.grid.get(*((pos, k) if axis == "x" else (k, pos))) is None)
    return axis, pos, gap


def crossing(episode, line):
    axis, pos, _ = line
    beyond = [(s[0] if axis == "x" else s[1]) > pos for s in episode["states"]]
    return {"crossed": any(beyond), "steps_to_cross": beyond.index(True) if any(beyond) else None}


def rollouts(run_dir):
    """Greedy and stochastic final-model episodes, checked against results/loop_metrics/."""
    model, normalize_obs, cfg = load_policy(run_dir)
    n, offset = cfg["eval"]["final_episodes"], cfg["eval"]["final_seed_offset"]
    result = {"greedy": run_episodes(model, cfg, n, offset, normalize_obs, True, record=True)}
    torch.manual_seed(TORCH_SEED)
    result["stochastic"] = run_episodes(model, cfg, n, offset, normalize_obs, False, record=True)
    for mode, episodes in result.items():
        reference = json.load(open(Path("results/loop_metrics") / f"{run_dir.name}_{mode}.json"))["episodes"]
        for key in ("success", "died", "time_limit", "revisit_ratio"):
            if [e[key] for e in episodes] != [e[key] for e in reference]:
                raise SystemExit(f"{run_dir.name} {mode}: re-run differs from results/loop_metrics in '{key}'")
    return cfg, result


def draw_path(ax, env, episode, title):
    """Map cells, the gap, and how often the agent stood on each cell (darker = more often)."""
    base = env.unwrapped
    ax.set_facecolor(SURFACE)
    visits = np.zeros((base.height, base.width))
    for x, y, _ in episode["states"]:
        visits[y, x] += 1
    for x in range(base.width):
        for y in range(base.height):
            cell = base.grid.get(x, y)
            if cell is not None:
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color=CELL_COLORS.get(cell.type, "#999999")))
            elif visits[y, x]:
                alpha = 0.15 + 0.85 * np.log1p(visits[y, x]) / np.log1p(visits.max())
                ax.add_patch(plt.Rectangle((x, y), 1, 1, color="#2a78d6", alpha=alpha, lw=0))
    xs = [s[0] + 0.5 for s in episode["states"]]
    ys = [s[1] + 0.5 for s in episode["states"]]
    ax.plot(xs[:1], ys[:1], marker="o", color=TEXT, ms=5)
    ax.set_xlim(0, base.width)
    ax.set_ylim(base.height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, fontsize=8, color=TEXT, loc="left")


def save_gif(cfg, episode, path):
    render_cfg = {**cfg, "env_kwargs": {**(cfg.get("env_kwargs") or {}), "render_mode": "rgb_array"}}
    env = make_env(render_cfg)
    env.reset(seed=episode["seed"])
    frames = [env.render()]
    for a in episode["actions"][:GIF_STEPS]:
        env.step(a)
        frames.append(env.render())
    env.close()
    iio.imwrite(path, np.stack(frames), duration=120, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dirs", nargs="+")
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    rows, examples = [], []
    for exp in args.exp_dirs:
        for run_dir in sorted(Path(exp).iterdir()):
            if not (run_dir / "final_eval_greedy.json").exists():
                continue
            cfg, result = rollouts(run_dir)
            env = make_env(cfg)
            row = {"run": run_dir.name, "seed": cfg["seed"]}
            for mode, episodes in result.items():
                for e in episodes:
                    env.reset(seed=e["seed"])
                    e["line"] = obstacle_line(env)
                    e.update(crossing(e, e["line"]))
                timeouts = [e for e in episodes if e["time_limit"]]
                successes = [e for e in episodes if e["success"]]
                row.update({
                    f"{mode}_timeouts": len(timeouts),
                    f"{mode}_timeouts_never_crossed": sum(not e["crossed"] for e in timeouts),
                    f"{mode}_succ_steps_to_cross": float(np.mean([e["steps_to_cross"] for e in successes]))
                    if successes else None,
                    f"{mode}_succ_steps_after_cross": float(np.mean([e["length"] - e["steps_to_cross"]
                                                                    for e in successes])) if successes else None,
                })
            rows.append(row)
            print(f"{run_dir.name:42s} " + "  ".join(
                f"{m}: {row[f'{m}_timeouts_never_crossed']}/{row[f'{m}_timeouts']} timeouts never crossed"
                for m in ("greedy", "stochastic")), flush=True)
            # Examples: the first timed-out episode of each LavaCrossing run, greedy and stochastic.
            if "Lava" in cfg["env_id"]:
                for mode, episodes in result.items():
                    timeout = next((e for e in episodes if e["time_limit"]), None)
                    if timeout is not None:
                        examples.append((cfg, run_dir.name, mode, timeout))

    table = Path("results/tables") / f"{args.name}_crossing.csv"
    with open(table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"table: {table}")

    # Figure: up to 8 example timeouts (alternating 7x7 / 5x5 runs where available).
    examples = sorted(examples, key=lambda x: (x[2], x[1]))[:8]
    fig, axes = plt.subplots(2, 4, figsize=(10, 6.2), facecolor=SURFACE)
    for ax in axes.flat:
        ax.axis("off")
    video_dir = Path("results/videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    for ax, (cfg, run, mode, e) in zip(axes.flat, examples):
        ax.axis("on")
        env = make_env(cfg)
        env.reset(seed=e["seed"])
        view = "5×5" if (cfg.get("env_kwargs") or {}).get("agent_view_size") == 5 else "7×7"
        side = "crossed the line" if e["crossed"] else "never crossed"
        draw_path(ax, env, e, f"{view} s{cfg['seed']}, {mode}, ep. seed {e['seed']}\n"
                              f"timeout, revisits {e['revisit_ratio']:.0%}, {side}")
        save_gif(cfg, e, video_dir / f"{run}_{mode}_ep{e['seed']}.gif")
    fig.suptitle("LavaCrossing timeouts: where the agent spent its 324 steps (blue: darker = more visits; "
                 "black dot: start; orange: lava; green: goal)", fontsize=9, color=TEXT_2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
    out = Path("results/figures") / f"{args.name}_paths.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"figure: {out}\nvideos: {video_dir}")


if __name__ == "__main__":
    main()

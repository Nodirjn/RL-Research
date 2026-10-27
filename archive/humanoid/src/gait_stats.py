"""Foot-contact and motion statistics of saved policies: what kind of movement produces the return?

Usage:
    python src/gait_stats.py experiments/EXP-002/C3_zoo_10M_s* --models 1000000 3000000 5000000 final \
        [--episodes 20] [--seed-offset 20000] --out results/tables/exp002_gait.csv

Per held-out episode (deterministic policy, same seeds as the final evaluation):
    airborne_frac        share of steps in which no body touches the floor (flight phases)
    left/right_contact   share of steps in which that foot touches the floor
    foot_asymmetry       (right - left) / (right + left) contact steps: 0 = symmetric, +1 = right foot only
    left/right_touchdowns number of times that foot lands (contact onsets); alternating steps give similar counts
    y_drift              sideways displacement (m), forward is +x
The printout aggregates episodes per (run, model) and then per model across runs.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from common import load_policy

FEET = ("left_foot", "right_foot")
COLUMNS = ("return", "length", "x_velocity", "y_drift", "airborne_frac", "left_contact", "right_contact",
           "foot_asymmetry", "left_touchdowns", "right_touchdowns")


def floor_bodies(model, data):
    """Names of the bodies currently touching the floor (world body 0)."""
    touching = set()
    for i in range(data.ncon):
        b1, b2 = model.geom_bodyid[data.contact[i].geom1], model.geom_bodyid[data.contact[i].geom2]
        if 0 in (b1, b2):
            touching.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1 or b2))
    return touching


def gait_episode(policy, normalize_obs, env, seed):
    obs, _ = env.reset(seed=seed)
    m, d = env.unwrapped.model, env.unwrapped.data
    x0, y0 = float(d.qpos[0]), float(d.qpos[1])
    ret, length, airborne = 0.0, 0, 0
    contact_steps = dict.fromkeys(FEET, 0)
    touchdowns = dict.fromkeys(FEET, 0)
    previous = set()
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = policy.predict(normalize_obs(obs) if normalize_obs is not None else obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        ret += float(reward)
        length += 1
        touching = floor_bodies(m, d)
        airborne += not touching
        for foot in FEET:
            contact_steps[foot] += foot in touching
            touchdowns[foot] += foot in touching and foot not in previous
        previous = touching
    total_contacts = sum(contact_steps.values())
    return {
        "return": ret,
        "length": length,
        "x_velocity": (float(d.qpos[0]) - x0) / (length * env.unwrapped.dt),
        "y_drift": float(d.qpos[1]) - y0,
        "airborne_frac": airborne / length,
        "left_contact": contact_steps["left_foot"] / length,
        "right_contact": contact_steps["right_foot"] / length,
        "foot_asymmetry": (contact_steps["right_foot"] - contact_steps["left_foot"]) / max(total_contacts, 1),
        "left_touchdowns": touchdowns["left_foot"],
        "right_touchdowns": touchdowns["right_foot"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--models", nargs="+", default=["final"], help="'final' and/or checkpoint steps")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-offset", type=int, default=20000)
    parser.add_argument("--out")
    args = parser.parse_args()

    rows = []
    for run_dir in args.run_dirs:
        for which in args.models:
            policy, normalize_obs, cfg = load_policy(run_dir, which)
            env = gym.make(cfg["env_id"])
            for i in range(args.episodes):
                rows.append({"run": Path(run_dir).name, "model": which, "seed": args.seed_offset + i,
                             **gait_episode(policy, normalize_obs, env, args.seed_offset + i)})
            env.close()

    per_run = defaultdict(list)
    for r in rows:
        per_run[(r["run"], r["model"])].append(r)
    run_means = {key: {c: np.mean([r[c] for r in eps]) for c in COLUMNS} for key, eps in per_run.items()}
    print("Per run (mean over episodes):")
    print("  " + " | ".join(("run", "model") + COLUMNS))
    for (run, which), means in run_means.items():
        print("  " + " | ".join([run, which] + [f"{means[c]:.2f}" for c in COLUMNS]))
    print("Per model (mean ± SD across runs):")
    for which in args.models:
        stats = [m for (_, w), m in run_means.items() if w == which]
        print(f"  {which}: " + "  ".join(f"{c}={np.mean([s[c] for s in stats]):.2f}±{np.std([s[c] for s in stats], ddof=1) if len(stats) > 1 else 0:.2f}"
                                         for c in COLUMNS))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["run", "model", "seed", *COLUMNS])
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()

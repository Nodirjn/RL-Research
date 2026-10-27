"""Train SB3 PPO from a YAML config, with periodic evaluation on fixed seeds and checkpoints.

Usage:
    python src/train.py --config configs/exp001_c2_cleanrl.yaml --seed 0 \
        --out experiments/EXP-001/C2_cleanrl_3M_s0 [--total-timesteps 1000000]

Writes to --out (refuses to overwrite an existing run):
    config.yaml          resolved config (incl. seed and CLI overrides)
    meta.json            software versions, host, command, wall-clock time, throughput
    eval.csv             one row per periodic evaluation (summary metrics)
    eval_episodes.csv    one row per evaluation episode
    final_eval.json      held-out evaluation of the final policy
    train_monitor.csv    every training episode (unnormalized return, length; with a `shaping` section also
                         the original return, the shaping return and the outcome)
    tb/                  TensorBoard logs
    final_model.zip, final_vecnormalize.pkl, checkpoints/
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from common import build_policy_kwargs, load_config, lr_schedule, make_env, run_episodes, software_versions, summarize
from shaping import EPISODE_KEYS, PBRS


class PeriodicEval(BaseCallback):
    """Every `eval.freq` env steps: evaluate the policy on fixed seeds; save checkpoints."""

    def __init__(self, cfg, run_dir):
        super().__init__()
        self.cfg = cfg
        self.deterministic = cfg["eval"].get("deterministic", True)
        self.freq = cfg["eval"]["freq"]
        self.n_episodes = cfg["eval"]["n_episodes"]
        self.seed_offset = cfg["eval"]["seed_offset"]
        self.checkpoints = sorted(cfg.get("checkpoints") or [])
        self.run_dir = run_dir
        self.next_eval = 0

    def _on_training_start(self):
        self.t0 = time.time()
        self.summary_file = open(self.run_dir / "eval.csv", "w", newline="")
        self.episode_file = open(self.run_dir / "eval_episodes.csv", "w", newline="")
        self.summary_writer = self.episode_writer = None
        self._evaluate()  # scheduled step 0 = the untrained policy

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self._evaluate()
        while self.checkpoints and self.num_timesteps >= self.checkpoints[0]:
            save_model(self.model, self.run_dir / "checkpoints" / f"step_{self.checkpoints.pop(0)}")
        return True

    def _on_training_end(self):
        self.summary_file.close()
        self.episode_file.close()

    def _evaluate(self):
        scheduled_step = self.next_eval
        self.next_eval += self.freq
        vecnorm = self.model.get_vec_normalize_env()
        episodes = run_episodes(self.model, self.cfg, self.n_episodes, self.seed_offset,
                                vecnorm.normalize_obs if vecnorm is not None else None, self.deterministic)
        row = {"scheduled_step": scheduled_step, "timesteps": self.num_timesteps,
               "wall_seconds": round(time.time() - self.t0, 1), **summarize(episodes)}

        if self.summary_writer is None:
            self.summary_writer = csv.DictWriter(self.summary_file, fieldnames=list(row))
            self.summary_writer.writeheader()
            self.episode_writer = csv.DictWriter(self.episode_file, fieldnames=["timesteps", *episodes[0]])
            self.episode_writer.writeheader()
        self.summary_writer.writerow(row)
        self.episode_writer.writerows({"timesteps": self.num_timesteps, **e} for e in episodes)
        self.summary_file.flush()
        self.episode_file.flush()

        for key in ("mean_return", "mean_length", "fall_rate", "mean_x_velocity", "success_rate"):
            if key in row:
                self.logger.record(f"eval/{key}", row[key])
        print(f"EVAL step={self.num_timesteps} return={row['mean_return']:.1f}+-{row['std_return']:.1f} "
              f"len={row['mean_length']:.0f} {outcome(row)} t={row['wall_seconds']:.0f}s", flush=True)


def outcome(summary):
    """The task's headline rate for log lines: falls (MuJoCo locomotion) or goal reached (MiniGrid)."""
    if "fall_rate" in summary:
        return f"fall_rate={summary['fall_rate']:.2f}"
    return f"success_rate={summary['success_rate']:.2f}"


def make_train_env(cfg):
    """The training env: make_env(cfg), wrapped in PBRS when the config has a `shaping` section.

    PBRS uses the PPO discount (PBRS rule 1). `shaping: {potential: none}` keeps the original reward and
    only adds the episode records. Evaluation builds its env with make_env, so it never sees shaping.
    """
    env = make_env(cfg)
    if cfg.get("shaping"):
        env = PBRS(env, gamma=cfg["ppo"]["gamma"], **cfg["shaping"])
    return env


def save_model(model, prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    model.save(f"{prefix}_model")
    vecnorm = model.get_vec_normalize_env()
    if vecnorm is not None:
        vecnorm.save(f"{prefix}_vecnormalize.pkl")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    # Overrides, for smoke tests and for runs that differ from the config only in budget.
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--eval-freq", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--final-episodes", type=int)
    parser.add_argument("--checkpoints", type=int, nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = args.total_timesteps
    if args.eval_freq is not None:
        cfg["eval"]["freq"] = args.eval_freq
    if args.eval_episodes is not None:
        cfg["eval"]["n_episodes"] = args.eval_episodes
    if args.final_episodes is not None:
        cfg["eval"]["final_episodes"] = args.final_episodes
    if args.checkpoints is not None:
        cfg["checkpoints"] = args.checkpoints
    cfg["seed"] = args.seed
    cfg["config_file"] = args.config

    run_dir = Path(args.out)
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists; refusing to overwrite a run")
    run_dir.mkdir(parents=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # One thread per run: several seeds run in parallel, and single-threaded CPU math is reproducible.
    torch.set_num_threads(1)
    meta = {"started": datetime.now().isoformat(timespec="seconds"), "command": " ".join(sys.argv),
            "versions": software_versions()}

    # VecMonitor sits inside VecNormalize, so train_monitor.csv logs the unnormalized reward: the original
    # one, or with shaping the shaped one (`r`) next to the PBRS episode records (EPISODE_KEYS).
    env = VecMonitor(DummyVecEnv([lambda: make_train_env(cfg)] * cfg["n_envs"]),
                     filename=str(run_dir / "train_monitor.csv"),
                     info_keywords=EPISODE_KEYS if cfg.get("shaping") else ())
    if cfg.get("normalize"):
        env = VecNormalize(env, **cfg["normalize"])

    ppo_kwargs = dict(cfg["ppo"])
    learning_rate = lr_schedule(ppo_kwargs.pop("learning_rate"), ppo_kwargs.pop("lr_schedule", "constant"))
    model = PPO("MlpPolicy", env, learning_rate=learning_rate,
                policy_kwargs=build_policy_kwargs(cfg.get("policy_kwargs")),
                seed=args.seed, device="cpu", tensorboard_log=str(run_dir / "tb"), verbose=0, **ppo_kwargs)

    start = time.time()
    model.learn(total_timesteps=cfg["total_timesteps"], callback=PeriodicEval(cfg, run_dir), tb_log_name="ppo")
    train_seconds = time.time() - start
    save_model(model, run_dir / "final")

    vecnorm = model.get_vec_normalize_env()
    episodes = run_episodes(model, cfg, cfg["eval"]["final_episodes"], cfg["eval"]["final_seed_offset"],
                            vecnorm.normalize_obs if vecnorm is not None else None, cfg["eval"].get("deterministic", True))
    final = {"timesteps": model.num_timesteps, "seed_offset": cfg["eval"]["final_seed_offset"],
             **summarize(episodes), "episodes": episodes}
    with open(run_dir / "final_eval.json", "w") as f:
        json.dump(final, f, indent=2)

    meta.update(finished=datetime.now().isoformat(timespec="seconds"), timesteps=model.num_timesteps,
                train_seconds=round(train_seconds, 1), steps_per_second=round(model.num_timesteps / train_seconds))
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"DONE {run_dir} steps={model.num_timesteps} final_return={final['mean_return']:.1f}+-{final['std_return']:.1f} "
          f"{outcome(final)} train_time={train_seconds / 60:.1f}min", flush=True)


if __name__ == "__main__":
    main()

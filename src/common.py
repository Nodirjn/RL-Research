"""Shared helpers: config loading, environment construction, PPO settings and policy evaluation.

Training-time and offline evaluation both go through `make_env` and `run_episodes`, so every
number in the project is measured by the same code.
"""
import importlib
import platform
import socket

import gymnasium as gym
import numpy as np
import torch.nn as nn
import yaml

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}
# Per-step reward terms reported by Humanoid-v5 in `info`; their sum is the env reward.
REWARD_KEYS = ("reward_survive", "reward_forward", "reward_ctrl", "reward_contact")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_env(cfg):
    """`gym.make(env_id, **env_kwargs)`, then each wrapper in `cfg["wrappers"]` (import paths), in order.

    Configs without `env_kwargs`/`wrappers` (the Humanoid experiments) get a plain `gym.make(env_id)`.
    Wrappers are imported before `gym.make`: importing their package (e.g. minigrid) registers its envs.
    """
    wrappers = []
    for path in cfg.get("wrappers") or []:
        module, name = path.rsplit(".", 1)
        wrappers.append(getattr(importlib.import_module(module), name))
    env = gym.make(cfg["env_id"], **(cfg.get("env_kwargs") or {}))
    for wrapper in wrappers:
        env = wrapper(env)
    return env


def budget_label(steps):
    """10000000 -> '10M', 500000 -> '500k': the budget part of run and condition names."""
    return f"{steps // 1_000_000}M" if steps >= 1_000_000 and steps % 1_000_000 == 0 else f"{steps // 1000}k"


def lr_schedule(initial, kind):
    """Constant LR, or linear annealing to 0 over training (CleanRL's `anneal_lr`)."""
    if kind == "constant":
        return initial
    if kind == "linear":
        # SB3 calls the schedule with progress_remaining, which goes 1 -> 0.
        return lambda progress_remaining: progress_remaining * initial
    raise ValueError(f"unknown lr_schedule {kind!r}")


def build_policy_kwargs(cfg):
    kwargs = dict(cfg or {})
    if "activation_fn" in kwargs:
        kwargs["activation_fn"] = ACTIVATIONS[kwargs["activation_fn"]]
    return kwargs


def load_policy(run_dir, which="final"):
    """Load a saved model ('final' or a checkpoint step) and its frozen observation normalizer.

    Returns (model, normalize_obs, cfg); normalize_obs is None for runs without VecNormalize.
    """
    from pathlib import Path

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    prefix = run_dir / "final" if which == "final" else run_dir / "checkpoints" / f"step_{which}"
    model = PPO.load(f"{prefix}_model.zip", device="cpu")
    normalize_obs = None
    if cfg.get("normalize"):
        vecnorm = VecNormalize.load(f"{prefix}_vecnormalize.pkl", DummyVecEnv([lambda: make_env(cfg)]))
        vecnorm.training = False
        normalize_obs = vecnorm.normalize_obs
    return model, normalize_obs, cfg


def minigrid_state(env):
    """The agent's cell and direction (x, y, dir). Within a Crossing episode the grid never changes,
    so this state fixes the observation."""
    base = env.unwrapped
    return int(base.agent_pos[0]), int(base.agent_pos[1]), int(base.agent_dir)


def minigrid_outcome(env, terminated, truncated):
    """Outcome flags of a finished MiniGrid episode, read from the env state, not from the reward.

    MiniGrid moves the agent onto the goal or lava cell that ends the episode, so the agent's cell
    tells the outcome. The reward sign cannot be used once shaping is added: a shaped step into
    lava can have a positive reward. A goal reached on the last allowed step counts as a success,
    not as a timeout, so the three flags are mutually exclusive.
    """
    base = env.unwrapped
    cell = base.grid.get(*base.agent_pos)
    reached_goal = cell is not None and cell.type == "goal"
    return {
        "success": bool(terminated and reached_goal),
        "died": bool(terminated and not reached_goal),     # e.g. stepped into lava
        "time_limit": bool(truncated and not terminated),  # ran out of steps
    }


def run_episodes(model, cfg, n_episodes, seed_offset, normalize_obs=None, deterministic=True, record=False):
    """Roll out the policy for `n_episodes` on seeds seed_offset, seed_offset+1, ...

    The env comes from `make_env(cfg)` with no reward wrappers, so returns are the original reward.
    `normalize_obs` applies frozen VecNormalize statistics (it never updates them).
    MuJoCo locomotion episodes record falls, forward motion and reward terms. MiniGrid episodes
    record the outcome (`minigrid_outcome`) and state revisits: `revisit_ratio` is the fraction of
    steps that end in an already-visited (x, y, dir) state (the start state counts as visited),
    `revisited` whether that happened at all. A greedy policy that revisits a state repeats
    itself until the time limit. With `record=True`, MiniGrid episodes also keep their `actions` and
    `states` (start state first), which replay the episode exactly from its seed.
    """
    env = make_env(cfg)
    mujoco_task = hasattr(env.unwrapped, "data")
    episodes = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + i)
        x0 = float(env.unwrapped.data.qpos[0]) if mujoco_task else None
        visited = set() if mujoco_task else {minigrid_state(env)}
        actions, states = [], [] if mujoco_task else [minigrid_state(env)]
        ret, length, revisits = 0.0, 0, 0
        components = dict.fromkeys(REWARD_KEYS, 0.0)
        terminated = truncated = False
        while not (terminated or truncated):
            policy_obs = normalize_obs(obs) if normalize_obs is not None else obs
            action, _ = model.predict(policy_obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ret += float(reward)
            length += 1
            if mujoco_task:
                for k in REWARD_KEYS:
                    components[k] += float(info[k])
            else:
                state = minigrid_state(env)
                revisits += state in visited
                visited.add(state)
                if record:
                    actions.append(int(action))
                    states.append(state)
        episode = {"episode": i, "seed": seed_offset + i, "return": ret, "length": length}
        if mujoco_task:
            x_displacement = float(env.unwrapped.data.qpos[0]) - x0
            episode.update({
                "fell": bool(terminated),          # unhealthy torso height -> terminated
                "time_limit": bool(truncated),     # reached the 1000-step limit
                "x_displacement": x_displacement,
                "mean_x_velocity": x_displacement / (length * env.unwrapped.dt),
                **components,
            })
        else:
            episode.update(minigrid_outcome(env, terminated, truncated))
            episode.update(revisit_ratio=revisits / length, revisited=revisits > 0)
            if record:
                episode.update(actions=actions, states=states)
        episodes.append(episode)
    env.close()
    return episodes


def summarize(episodes):
    returns = np.array([e["return"] for e in episodes])
    summary = {
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "mean_length": float(np.mean([e["length"] for e in episodes])),
    }
    if "fell" in episodes[0]:
        summary.update(
            fall_rate=float(np.mean([e["fell"] for e in episodes])),
            mean_x_displacement=float(np.mean([e["x_displacement"] for e in episodes])),
            mean_x_velocity=float(np.mean([e["mean_x_velocity"] for e in episodes])),
        )
    else:
        summary.update(
            success_rate=float(np.mean([e["success"] for e in episodes])),
            death_rate=float(np.mean([e["died"] for e in episodes])),
            timeout_rate=float(np.mean([e["time_limit"] for e in episodes])),
            loop_rate=float(np.mean([e["revisited"] for e in episodes])),
            revisit_ratio=float(np.mean([e["revisit_ratio"] for e in episodes])),
        )
    return summary


def software_versions():
    import mujoco
    import stable_baselines3
    import torch

    return {
        "python": platform.python_version(),
        "gymnasium": gym.__version__,
        "mujoco": mujoco.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }

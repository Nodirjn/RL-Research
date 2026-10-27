"""Outcome detection and revisit counting for MiniGrid episodes (src/common.py).

Run from the project root: python -m pytest tests
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import make_env, minigrid_outcome, minigrid_state, run_episodes  # noqa: E402

FORWARD, LEFT = 2, 0
SIMPLE = {"env_id": "MiniGrid-SimpleCrossingS9N1-v0", "wrappers": ["minigrid.wrappers.FlatObsWrapper"]}
LAVA = {"env_id": "MiniGrid-LavaCrossingS9N1-v0", "wrappers": ["minigrid.wrappers.FlatObsWrapper"]}


def place_facing(env, target_type):
    """Put the agent on a free cell next to a `target_type` cell, facing it."""
    from minigrid.core.constants import DIR_TO_VEC

    base = env.unwrapped
    for x in range(base.width):
        for y in range(base.height):
            cell = base.grid.get(x, y)
            if cell is None or cell.type != target_type:
                continue
            for d, vec in enumerate(DIR_TO_VEC):
                nx, ny = x - vec[0], y - vec[1]
                if base.grid.get(nx, ny) is None:
                    base.agent_pos, base.agent_dir = (nx, ny), d
                    return
    raise AssertionError(f"no free cell next to a {target_type} cell")


def test_goal_step_is_success():
    env = make_env(SIMPLE)
    env.reset(seed=0)
    place_facing(env, "goal")
    _, reward, terminated, truncated, _ = env.step(FORWARD)
    assert terminated and not truncated and reward > 0
    assert minigrid_outcome(env, terminated, truncated) == {"success": True, "died": False, "time_limit": False}


def test_lava_step_is_death_even_with_positive_reward():
    # A shaped reward can be positive on the fatal step; the outcome must still be a death.
    env = gym.wrappers.TransformReward(make_env(LAVA), lambda r: r + 1.0)
    env.reset(seed=0)
    place_facing(env, "lava")
    _, reward, terminated, truncated, _ = env.step(FORWARD)
    assert terminated and reward == pytest.approx(1.0)
    assert minigrid_outcome(env, terminated, truncated) == {"success": False, "died": True, "time_limit": False}


def test_time_limit_is_timeout():
    env = make_env(SIMPLE)
    env.reset(seed=0)
    env.unwrapped.step_count = env.unwrapped.max_steps - 1
    _, _, terminated, truncated, _ = env.step(LEFT)
    assert truncated and not terminated
    assert minigrid_outcome(env, terminated, truncated) == {"success": False, "died": False, "time_limit": True}


def test_goal_on_last_step_is_success_not_timeout():
    env = make_env(SIMPLE)
    env.reset(seed=0)
    place_facing(env, "goal")
    env.unwrapped.step_count = env.unwrapped.max_steps - 1
    _, _, terminated, truncated, _ = env.step(FORWARD)
    assert terminated and truncated
    assert minigrid_outcome(env, terminated, truncated) == {"success": True, "died": False, "time_limit": False}


class ConstantPolicy:
    def __init__(self, action):
        self.action = action

    def predict(self, obs, deterministic=True):
        return self.action, None


class RandomPolicy:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=True):
        return int(self.rng.integers(0, 3)), None  # left, right, forward


def test_turning_in_place_revisits_every_state_after_the_first_turn_cycle():
    episodes = run_episodes(ConstantPolicy(LEFT), SIMPLE, n_episodes=2, seed_offset=0)
    max_steps = make_env(SIMPLE).unwrapped.max_steps
    for e in episodes:
        assert e["length"] == max_steps and e["time_limit"] and not e["success"]
        # Turns 1-3 reach new directions; from turn 4 on every state was seen before.
        assert e["revisited"] and e["revisit_ratio"] == pytest.approx((max_steps - 3) / max_steps)


def test_outcome_matches_the_reward_on_the_unshaped_env():
    # Without shaping the old rule (reward > 0 = success) is correct, so both must agree.
    max_steps = make_env(LAVA).unwrapped.max_steps
    episodes = run_episodes(RandomPolicy(0), LAVA, n_episodes=30, seed_offset=0)
    assert any(e["died"] for e in episodes)
    for e in episodes:
        assert e["success"] == (e["return"] > 0)
        assert e["died"] == (e["return"] == 0 and e["length"] < max_steps)
        assert e["time_limit"] == (e["return"] == 0 and e["length"] == max_steps)
        assert e["success"] + e["died"] + e["time_limit"] == 1


def test_state_is_cell_and_direction():
    env = make_env(SIMPLE)
    env.reset(seed=0)
    x, y, d = minigrid_state(env)
    env.step(LEFT)
    assert minigrid_state(env) == (x, y, (d - 1) % 4)


def test_recorded_actions_replay_the_episode():
    episodes = run_episodes(RandomPolicy(1), LAVA, n_episodes=5, seed_offset=100, record=True)
    env = make_env(LAVA)
    for e in episodes:
        env.reset(seed=e["seed"])
        states = [minigrid_state(env)]
        for a in e["actions"]:
            env.step(a)
            states.append(minigrid_state(env))
        assert states == e["states"] and len(e["actions"]) == e["length"]

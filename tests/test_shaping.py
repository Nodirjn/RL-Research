"""PBRS wrapper (src/shaping.py): the formula, terminal and truncation handling, potentials, episode records.

Run from the project root: python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import make_env  # noqa: E402
from shaping import PBRS, bfs_distances  # noqa: E402
from common import run_episodes  # noqa: E402
from test_minigrid_outcome import FORWARD, LAVA, LEFT, SIMPLE, RandomPolicy, place_facing  # noqa: E402

GAMMA = 0.99
D = np.hypot(6, 6)  # start (1, 1) to goal (7, 7)


def shaped(cfg, seed=0, **kwargs):
    env = PBRS(make_env(cfg), gamma=kwargs.pop("gamma", GAMMA), **kwargs)
    env.reset(seed=seed)
    return env


def euclidean_phi(env):
    x, y = env.unwrapped.agent_pos
    return -np.hypot(x - 7, y - 7) / D


def teleport_facing(env, target_type):
    """Put the agent next to a goal/lava cell, facing it, and re-read Phi of the new position."""
    place_facing(env, target_type)
    env.last_phi = env.phi()


def test_euclidean_potential_is_minus_one_at_the_start():
    env = shaped(SIMPLE)
    assert env.scale == pytest.approx(8.485, abs=1e-3)
    assert env.last_phi == pytest.approx(-1.0)


def test_every_nonterminal_step_gets_gamma_phi_next_minus_phi():
    env = shaped(SIMPLE)
    rng = np.random.default_rng(0)
    for _ in range(60):
        phi = euclidean_phi(env)
        _, reward, terminated, truncated, _ = env.step(int(rng.integers(0, 3)))
        if terminated or truncated:
            break
        assert reward == pytest.approx(GAMMA * euclidean_phi(env) - phi)


def test_turning_in_place_gives_the_small_gamma_bonus():
    env = shaped(SIMPLE)
    _, reward, *_ = env.step(LEFT)
    assert reward == pytest.approx((GAMMA - 1) * -1.0)  # +0.01 at the start, d/D = 1


def test_goal_step_uses_zero_terminal_potential():
    env = shaped(SIMPLE)
    teleport_facing(env, "goal")
    phi = env.last_phi
    _, reward, terminated, _, info = env.step(FORWARD)
    original = info["original_return"]
    assert terminated and info["outcome"] == "success" and original > 0
    assert reward == pytest.approx(original - phi)  # F = gamma * 0 - Phi(s)


def test_lava_step_gets_positive_shaping_and_still_counts_as_death():
    # H-lava-a: with Phi(terminal) = 0, the fatal step gets F = -Phi(s) = d/D > 0.
    env = shaped(LAVA)
    teleport_facing(env, "lava")
    phi = env.last_phi
    _, reward, terminated, _, info = env.step(FORWARD)
    assert terminated and info["outcome"] == "death" and info["original_return"] == 0
    assert reward == pytest.approx(-phi) and reward > 0


def test_truncation_uses_the_normal_formula():
    env = shaped(SIMPLE)
    env.unwrapped.step_count = env.unwrapped.max_steps - 1
    phi = env.last_phi
    _, reward, terminated, truncated, info = env.step(LEFT)
    assert truncated and not terminated and info["outcome"] == "timeout"
    assert reward == pytest.approx((GAMMA - 1) * phi)  # not -phi: the state is not terminal


def test_episode_records_and_telescoping_with_gamma_one():
    # With gamma = 1 the shaping terms telescope: their sum is Phi(last) - Phi(start).
    env = shaped(SIMPLE, gamma=1.0)
    plain = make_env(SIMPLE)
    plain.reset(seed=0)
    rng = np.random.default_rng(3)
    total, original = 0.0, 0.0
    while True:
        action = int(rng.integers(0, 3))
        _, reward, terminated, truncated, info = env.step(action)
        _, plain_reward, *_ = plain.step(action)
        total += reward
        original += plain_reward
        if terminated or truncated:
            break
    last_phi = 0.0 if terminated else euclidean_phi(env)
    assert info["shaping_return"] == pytest.approx(last_phi - (-1.0))
    assert info["original_return"] == pytest.approx(original)
    assert total == pytest.approx(original + info["shaping_return"])


def test_reset_starts_a_new_episode_record():
    env = shaped(SIMPLE)
    env.step(LEFT)
    env.reset(seed=1)
    assert env.last_phi == pytest.approx(-1.0) and env.original_return == env.shaping_return == 0.0


@pytest.mark.parametrize("potential", ["euclidean", "bfs"])
def test_discounted_shaping_return_of_a_terminated_episode_is_one(potential):
    # Phi(start) = -1 and Phi(terminal) = 0, so sum_t gamma^t F_t = gamma^T * 0 - Phi(start) = 1 for every
    # episode that ends at the goal or in lava, under both distance potentials (PBRS rule 3, note).
    rng = np.random.default_rng(0)
    terminated_episodes = 0
    for seed in range(20):
        env = shaped(LAVA, seed=seed, potential=potential)
        plain = make_env(LAVA)
        plain.reset(seed=seed)
        discounted, t = 0.0, 0
        while True:
            action = int(rng.integers(0, 3))
            _, reward, terminated, truncated, _ = env.step(action)
            _, plain_reward, *_ = plain.step(action)
            discounted += GAMMA ** t * (reward - plain_reward)
            t += 1
            if terminated or truncated:
                break
        if terminated:
            terminated_episodes += 1
            assert discounted == pytest.approx(1.0)
    assert terminated_episodes > 0


def test_bfs_potential_follows_the_path_through_the_gap():
    env = shaped(LAVA, potential="bfs")
    base = env.unwrapped
    dist = bfs_distances(base.grid, (7, 7))
    assert env.scale == dist[1, 1] >= 12  # at least the Manhattan distance
    lava = [(x, y) for x in range(9) for y in range(9) if base.grid.get(x, y) and base.grid.get(x, y).type == "lava"]
    assert lava and all(dist[c] == np.inf for c in lava)
    # Walking down the BFS distance from the start reaches the goal through the gap, one cell per step.
    from behavior import obstacle_line

    _, _, gap = obstacle_line(env)
    path = [(1, 1)]
    while path[-1] != (7, 7):
        x, y = path[-1]
        path.append(min(((x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))), key=lambda c: dist[c]))
        assert dist[path[-1]] == dist[path[-2]] - 1
    assert gap in path and len(path) - 1 == env.scale


def test_euclidean_potential_peaks_at_the_corner_nearest_the_goal_not_at_the_gap():
    # The behavior-analysis argument: on the start side of a vertical line at x = c, Euclidean Phi is
    # highest at (c - 1, 7), which is a dead end unless the gap is in the last row.
    from behavior import obstacle_line

    for seed in range(200):
        env = shaped(LAVA, seed=seed)
        axis, c, gap = obstacle_line(env)
        if axis == "x" and gap[1] != 7:
            break
    base = env.unwrapped
    start_side = [(x, y) for x in range(1, c) for y in range(1, 8) if base.grid.get(x, y) is None]
    phis = {}
    for cell in start_side:
        base.agent_pos = cell
        phis[cell] = env.phi()
    assert max(phis, key=phis.get) == (c - 1, 7) and (c - 1, 7) != (gap[0] - 1, gap[1])


def test_lava_penalty_lowers_phi_next_to_lava_only():
    env = shaped(LAVA, lava_penalty=0.1)
    reference = shaped(LAVA)
    place_facing(env, "lava")
    place_facing(reference, "lava")
    assert env.phi() == pytest.approx(reference.phi() - 0.1)
    env.reset(seed=0)
    reference.reset(seed=0)
    # The start (1, 1) is never next to lava: S9N1 lines are at x or y in {2, 4, 6}, but (2, 1) can be lava.
    start_next_to_lava = any(
        (c := env.unwrapped.grid.get(1 + dx, 1 + dy)) is not None and c.type == "lava"
        for dx, dy in ((1, 0), (0, 1)))
    assert env.phi() == pytest.approx(reference.phi() - (0.1 if start_next_to_lava else 0.0))


def test_offset_shifts_phi():
    env = shaped(SIMPLE, offset=1.0)
    assert env.last_phi == pytest.approx(0.0)  # 1 - d/D at the start


def test_no_potential_leaves_the_reward_unchanged_but_records_the_episode():
    env = shaped(SIMPLE, potential="none")
    teleport_facing(env, "goal")
    _, reward, terminated, _, info = env.step(FORWARD)
    assert terminated and reward == info["original_return"] > 0 and info["shaping_return"] == 0.0


def test_naive_control_pays_for_going_back_and_forth():
    # Ng et al. (1999): a bonus for progress with no cost for moving away has positive cycles.
    # Forward, turn around, forward back, turn around: the agent is where it started, facing the same way.
    for seed in range(50):
        naive, pbrs = shaped(SIMPLE, seed=seed, naive=True), shaped(SIMPLE, seed=seed, gamma=1.0)
        if naive.unwrapped.grid.get(2, 1) is None:  # the cell ahead of the start is free
            break
    start = tuple(naive.unwrapped.agent_pos), naive.unwrapped.agent_dir
    progress = None
    totals = []
    for env in (naive, pbrs):
        total = 0.0
        for action in [FORWARD, LEFT, LEFT, FORWARD, LEFT, LEFT] * 3:
            phi = euclidean_phi(env)
            _, reward, *_ = env.step(action)
            if progress is None and euclidean_phi(env) != phi:
                progress = euclidean_phi(env) - phi
            total += reward
        totals.append(total)
    assert (tuple(naive.unwrapped.agent_pos), naive.unwrapped.agent_dir) == start
    assert progress > 0 and totals[0] == pytest.approx(3 * progress)  # paid once per cycle
    assert totals[1] == pytest.approx(0.0)  # PBRS: a cycle earns nothing (gamma = 1)


def test_naive_control_uses_the_cell_it_ends_on_even_at_termination():
    env = shaped(LAVA, naive=True)
    teleport_facing(env, "lava")
    phi = env.last_phi
    _, reward, terminated, _, info = env.step(FORWARD)
    assert terminated and info["outcome"] == "death"
    assert reward == pytest.approx(max(0.0, euclidean_phi(env) - phi))  # the agent stands on the lava cell
    assert reward < -phi  # below the +d/D that the PBRS convention Phi(terminal) = 0 would pay


def test_evaluation_never_sees_shaping():
    # train.py wraps only the training env; evaluation (run_episodes, load_policy, evaluate.py) uses make_env.
    from train import make_train_env

    cfg = {**LAVA, "shaping": {"potential": "euclidean"}, "ppo": {"gamma": GAMMA}}
    assert isinstance(make_train_env(cfg), PBRS)
    layer = make_env(cfg)
    while hasattr(layer, "env"):
        assert not isinstance(layer, PBRS)
        layer = layer.env
    shaped_cfg = run_episodes(RandomPolicy(0), cfg, n_episodes=20, seed_offset=0)
    plain_cfg = run_episodes(RandomPolicy(0), LAVA, n_episodes=20, seed_offset=0)
    assert [e["return"] for e in shaped_cfg] == [e["return"] for e in plain_cfg]
    assert all(e["return"] > 0 if e["success"] else e["return"] == 0 for e in shaped_cfg)

"""Potential-based reward shaping (PBRS) for MiniGrid navigation, plus a non-potential negative control.
Used by train.py only: evaluation
(common.run_episodes) builds its env with make_env and always sees the original reward.

    F(s, s') = gamma * Phi(s') - Phi(s),  with Phi(s') = 0 when s' is terminal (goal or lava).

On truncation (time limit) s' is not terminal and the normal formula is used, matching SB3, which
bootstraps V(s') at truncation. Zero terminal potentials keep the optimal policy unchanged in episodic
tasks (Grzes 2017, AAMAS). gamma is the PPO discount (train.py passes cfg["ppo"]["gamma"]).

Potentials (the agent's cell p, the goal g; distances in cells):
    euclidean   Phi(p) = offset - ||p - g||_2 / D_E,  D_E = ||start - g||_2 (8.485 on S9 maps)
    bfs         Phi(p) = offset - bfs(p, g) / D_B,    D_B = bfs(start, g); shortest 4-neighbour path over
                cells the agent can stand on without dying (no walls, no lava)
    none        no distance term (Phi = 0 unless lava_penalty is set): with no lava penalty either, the
                reward is unchanged and only the episode records below are added
    lava_penalty c > 0 adds -c to Phi(p) when a 4-neighbour of p is lava (the professor's lava term)
With offset = 0, Phi(start) = -1 and Phi rises to about -1/D next to the goal.

naive = True is the negative control, not PBRS: F = max(0, Phi(s') - Phi(s)), a bonus for every step that
gets closer to the goal and nothing for a step away. Moving back and forth then earns reward (the positive
cycles of Ng et al. 1999, the bicycle example). Phi(s') is the potential of the cell the agent ends on, also
when the episode terminates: with the PBRS convention Phi(terminal) = 0, entering lava would earn +d/D, and
the control would test that convention instead of the missing potential form.

Every episode's last `info` carries `outcome` (success / death / timeout, from the env state),
`original_return` and `shaping_return` (the sum of F), so train_monitor.csv separates the original and
the shaped reward (train.py passes these keys to VecMonitor).
"""
from collections import deque

import gymnasium as gym
import numpy as np

from common import minigrid_outcome

EPISODE_KEYS = ("outcome", "original_return", "shaping_return")
OUTCOME_NAMES = {"success": "success", "died": "death", "time_limit": "timeout"}
NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def is_lava(grid, x, y):
    cell = grid.get(x, y)
    return cell is not None and cell.type == "lava"


def bfs_distances(grid, goal):
    """Shortest 4-neighbour path length from every reachable cell to the goal, avoiding walls and lava."""
    dist = np.full((grid.width, grid.height), np.inf)
    dist[goal] = 0
    queue = deque([goal])
    while queue:
        x, y = queue.popleft()
        for dx, dy in NEIGHBOURS:
            nx, ny = x + dx, y + dy
            cell = grid.get(nx, ny)
            if (cell is None or cell.type == "goal") and dist[nx, ny] == np.inf:
                dist[nx, ny] = dist[x, y] + 1
                queue.append((nx, ny))
    return dist


class PBRS(gym.Wrapper):
    def __init__(self, env, potential="euclidean", gamma=0.99, offset=0.0, lava_penalty=0.0, naive=False):
        super().__init__(env)
        if potential not in ("euclidean", "bfs", "none"):
            raise ValueError(f"unknown potential {potential!r}")
        self.potential, self.gamma, self.offset, self.lava_penalty = potential, gamma, offset, lava_penalty
        self.naive = naive

    def phi(self):
        """Phi of the agent's current cell (the potential of a non-terminal state)."""
        base = self.env.unwrapped
        x, y = (int(v) for v in base.agent_pos)
        if self.potential == "euclidean":
            value = self.offset - np.hypot(x - self.goal[0], y - self.goal[1]) / self.scale
        elif self.potential == "bfs":
            value = self.offset - self.bfs[x, y] / self.scale
        else:
            value = 0.0
        if self.lava_penalty and any(is_lava(base.grid, x + dx, y + dy) for dx, dy in NEIGHBOURS):
            value -= self.lava_penalty
        return float(value)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        base = self.env.unwrapped
        self.goal = tuple(int(v) for v in base.goal_position)
        start = tuple(int(v) for v in base.agent_pos)
        if self.potential == "euclidean":
            self.scale = float(np.hypot(start[0] - self.goal[0], start[1] - self.goal[1]))
        elif self.potential == "bfs":
            self.bfs = bfs_distances(base.grid, self.goal)
            self.scale = float(self.bfs[start])
        self.last_phi = self.phi()
        self.original_return = self.shaping_return = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.naive:
            next_phi = self.phi()
            shaping = max(0.0, next_phi - self.last_phi)
        else:
            next_phi = 0.0 if terminated else self.phi()
            shaping = self.gamma * next_phi - self.last_phi
        self.last_phi = next_phi
        self.original_return += float(reward)
        self.shaping_return += shaping
        if terminated or truncated:
            outcome = minigrid_outcome(self.env, terminated, truncated)
            info = {**info, "outcome": next(OUTCOME_NAMES[k] for k in OUTCOME_NAMES if outcome[k]),
                    "original_return": self.original_return, "shaping_return": self.shaping_return}
        return obs, float(reward) + shaping, terminated, truncated, info

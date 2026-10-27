# Stage 3 — Change of direction: from Humanoid to MiniGrid

**Workflow step:** between steps 2 and 3 (choice of the study environment). **Status:** done.

## Question

Does MuJoCo Humanoid fit the research idea, or should the study move to a lightweight grid-world
environment?

## Why this stage

The research idea is a reward-shaping study with:

- PPO as the baseline;
- lightweight Gymnasium environments, such as **MiniGrid or Safety-Gymnasium**, that allow thousands of
  episodes on a laptop, because physics simulation adds complexity unrelated to reward shaping;
- limited visual observations, i.e. a restricted egocentric view such as 7×7 or 5×5;
- a move from a sparse goal reward to a dense potential-based reward, e.g. distance to the goal or penalties
  near obstacles.

The Humanoid work ([stage 2](02_humanoid_baseline.md)) did not match this:

| Requirement | Humanoid-v5 |
|---|---|
| Navigation | Locomotion |
| Limited view | Full proprioceptive state |
| Sparse reward | Already dense (survival + forward velocity) |
| Lightweight | Computationally expensive on a CPU-only laptop |

Its baseline also did not walk: most of the effort would have gone into locomotion, not into reward shaping.

## What was done

- **Compatibility check.** minigrid 3.0.0 works with the project's Python 3.9 and gymnasium 1.0.0 (newer
  minigrid versions need Python ≥ 3.10). safety-gymnasium 1.0.0 pins older gymnasium, mujoco and pygame
  versions and would need a separate environment. MiniGrid was therefore the cheaper choice.
- **Installation:** only `minigrid==3.0.0` was added; gymnasium stayed at 1.0.0.
- **Smoke test with a random policy** (to see how hard the candidate tasks are):

| Task | Grid | Step limit | Random-policy success |
|---|---|---|---|
| SimpleCrossingS9N1 | 9×9 | 324 | 1/26 episodes |
| LavaCrossingS9N1 | 9×9 | 324 | 1/58 |
| FourRooms | 19×19 | 100 | 2/55 |
| DoorKey-8x8 | 8×8 | 640 | 0/13 |

- **Agent settings taken from a published source:** the RL Zoo `minigrid-defaults` for PPO (`FlatObsWrapper`,
  8 environments, n_steps 128, batch 64, γ 0.99, λ 0.95, 10 epochs, entropy 0, lr 2.5e-4, clip 0.2,
  normalization on).
- **Code changes:** `make_env` builds any environment and wrapper list from the config; `run_episodes`
  records success, death and timeout for MiniGrid; `evaluate.py` gained a stochastic mode.
- **The Humanoid path was verified unchanged** after these changes: a re-evaluation of a saved Humanoid model
  reproduced its recorded evaluation exactly, and a short re-training reproduced the first recorded
  evaluation rows exactly.

## Decisions and reasons

- **Move to MiniGrid and keep Humanoid as a preliminary study.** Reason: MiniGrid meets all four
  requirements above, and Humanoid meets none of them. Tuning and compute for Humanoid would serve
  locomotion, not the study of reward shaping.
- **Tasks: SimpleCrossingS9N1 and LavaCrossingS9N1.** Reason: both are navigation around an obstacle line
  with one gap and a sparse reward, and they differ in one point: a wall is harmless, lava ends the episode.
  This contrast between plain navigation and a high-risk obstacle strengthens the comparison.
- **Method direction:** PBRS with a distance potential (Φ is minus the straight-line distance from the agent
  to the goal, so it rises as the agent gets closer) and, on LavaCrossing, an extra negative potential on
  cells next to lava (risk aversion). Reason: these are the two most direct ways to make the sparse goal
  reward dense: progress toward the goal, and caution near the deadly obstacle. A full baseline with 5 seeds
  on both tasks comes first, so that every shaping result has a reference. The exact formulas are in the
  [research plan](../research_plan.md#53-shaping).
- **The RL Zoo defaults are used unchanged.** Reason: a published configuration avoids tuning the baseline
  toward one outcome. The Crossing tasks are not in RL Zoo's tuned list, so this is also a limitation.
- **Humanoid files were moved to `archive/humanoid/`, not deleted**, so that the working folders hold only
  the MiniGrid study while the preliminary work stays reproducible.

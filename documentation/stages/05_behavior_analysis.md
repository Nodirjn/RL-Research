# Stage 5 — Behavior analysis: how and where the baseline fails

**Workflow step:** 3 (behavior analysis). **Status:** done for the final policies. The analysis covers the
final policies of both EXP-011 (7×7 view) and EXP-012 (5×5 view, [stage 6](06_view_size.md)); EXP-012 had
finished by the time the analysis ran, so both are reported together.

## Question

The baseline's main failure on LavaCrossing is the time limit ([stage 4](04_minigrid_baseline.md)). Do these
timeouts come from looping, or from not finding the gap? And where in the map does the agent fail?

## Why this stage

High reward is not proof of the intended behavior, and a success rate does not say how the agent fails. The
planned shaping conditions use a distance potential and a lava potential. Before the conditions are fixed,
the failure they should address has to be identified from behavior, so that the expected effect of each
potential can be stated in advance.

## Step 1: reliable outcome detection

**What:** `common.minigrid_outcome` now reads the agent's cell instead of the reward sign: success =
terminated on the goal; death = terminated anywhere else (lava); timeout = truncated and not terminated.
Seven tests in `tests/test_minigrid_outcome.py` check it.

**Why:** with shaping, the reward of a lava step can be positive, and a reward-sign rule would count it as a
success. The tests include exactly that case. A goal reached on the last allowed step was previously counted
as both success and timeout; it is now only a success. No reported number changed: the re-run below
reproduces all 2000 recorded greedy outcomes.

## Step 2: loop metrics (pre-registered addendum)

**Definitions.** A state is the agent's cell and direction (x, y, dir).

- **Greedy loop rate:** fraction of greedy episodes in which the agent revisits a state. The map does not
  change within an episode and the observation normalizer is frozen, so a revisited state gives the same
  observation and the greedy policy repeats itself until the time limit.
- **Revisit ratio:** for stochastic episodes, the fraction of steps that end in an already visited state.

**How:** `src/loop_metrics.py` re-runs every final model on the same 100 held-out episodes, greedy and
stochastic (torch seed 0). Check: for all 20 runs, the greedy re-run reproduces the recorded greedy
evaluation episode by episode.

| Metric (mean over 5 seeds) | Simple 7×7 | Simple 5×5 | Lava 7×7 | Lava 5×5 |
|---|---|---|---|---|
| Greedy loop rate | 0.026 | 0.140 | 0.380 | 0.244 |
| Greedy timeouts that looped | 100% | 100% | 100% | 100% |
| Greedy episodes that looped and still succeeded | 0 | 0 | 0 | 0 |
| Revisit ratio, stochastic, all episodes | 0.010 | 0.079 | 0.278 | 0.170 |
| — successful episodes | 0.010 | 0.052 | 0.126 | 0.086 |
| — timed-out episodes ‡ | no timeouts | 0.860 | 0.877 | 0.883 |

‡ Over the seeds that had stochastic timeouts (2 seeds for SimpleCrossing 5×5, 4 for LavaCrossing 7×7).

A timed-out stochastic episode visits on average only 34–42 of the 172 possible (cell, direction) states.

**Observations:**

1. Every greedy failure is a loop: each greedy timeout revisited a state, and no looping greedy episode
   reached the goal.
2. Timed-out stochastic episodes spend 86–88% of their steps in states already visited: the agent stays in
   a small part of the map for 324 steps.
3. The weakest run (LavaCrossing 7×7, seed 2) has the highest revisit ratio (0.78 over all episodes).

## Step 3: where does the agent fail? (line-crossing analysis)

**How:** `src/behavior.py` replays the same episodes with trajectories recorded. An S9N1 map has one
obstacle line with one gap, and the goal is always behind it. "Crossed" means the agent stood on the goal side
of the line at some step. Outputs: `results/tables/exp011_exp012_crossing.csv`, the path figure
`results/figures/exp011_exp012_paths.png` (8 example LavaCrossing timeouts) and GIFs of the same episodes in
`results/videos/`.

Timed-out episodes that **never crossed** the line (pooled over 5 seeds):

| Condition | Greedy | Stochastic |
|---|---|---|
| LavaCrossing 7×7 | **165/190 = 0.87** | **102/116 = 0.88** |
| LavaCrossing 5×5 | 71/122 = 0.58 | 31/53 = 0.58 |
| SimpleCrossing 7×7 | 13/13 | no timeouts |
| SimpleCrossing 5×5 | 46/70 = 0.66 | 16/17 |

**Observations:**

1. With a 7×7 view, nearly all LavaCrossing timeouts happen before the line. Seed 2 never crosses in 70 of
   70 stochastic timeouts.
2. The example paths show two typical patterns: the agent walks into a corner next to the line and turns back
   and forth there (98–99% revisits, ≤ 3 cells), or it runs a fixed lap through the first room that never
   passes the gap.
3. With a 5×5 view, about 40% of LavaCrossing timeouts happen after crossing: the agent gets through the gap
   but does not reach the goal. A plausible reason (not tested) is that with 5×5 the goal can still be out of
   view after crossing, while with 7×7 it is often visible.

## Interpretations

- **Answer to the question:** on LavaCrossing with a 7×7 view, timeouts come mainly from not getting through
  the gap, and the agent loops while failing. Both explanations hold together: the loop happens on the start
  side of the line.
- This fits a partial-observability explanation: the memoryless policy acts on a local view, and when that
  view looks the same in several places, it repeats itself. A potential on the global position prefers states
  closer to the goal and could break such cycles. This is a hypothesis for the shaping experiments, not a
  result.
- **Consequence for the Euclidean potential (H-dead-end):** this potential grows as the straight-line
  distance to the goal shrinks (Φ = −d/D, see the [notation](../research_plan.md#52-notation)). On the start
  side of the line it is therefore highest at the cell next to the line that is closest to the goal, not at
  the gap, and it can pull the agent into a dead end. A path-length (BFS) potential has no such dead end.
  The BFS potential was already planned as an extension; this analysis gives a concrete reason for it.
- **Consequence for the lava-proximity potential (H-lava-b):** the gap lies in the lava line itself, so the
  agent must stand next to lava to cross. A penalty near lava may make the main failure, not crossing, worse.
  This expectation is stated before the condition is run.
- A recurrent (memory-based) policy is another known remedy for loops. It is not in the plan because it
  changes the agent, not the reward; it is an alternative explanation to discuss in the paper.

## Decisions and reasons

- **Loop metrics and the never-crossed fraction become secondary metrics for every shaping condition**, and
  both scripts run on every final model. Reason: they measure exactly the failure that shaping should fix.
- **The two shaping expectations above are written into the plan as hypotheses (H-dead-end, H-lava-b), not
  as results.**
- **Intermediate checkpoints (1M, 2M) are not analyzed now**; they are kept in case the paper needs them.

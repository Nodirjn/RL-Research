# Stage 4 — MiniGrid pilot and baseline

**Workflow step:** 2 (baseline). **Experiments:** EXP-010 (pilot), EXP-011 (baseline). **Status:** done.

## Question

How sample-efficient is standard PPO with the original sparse reward on SimpleCrossing and LavaCrossing?
This baseline is the reference for every shaping condition later.

## EXP-010: pilot

### Why

A pilot is not a result. It checks the pipeline and the computational cost, shows the shape of the learning
curves, and tells which evaluation mode is informative, so that the real baseline can be designed and
pre-registered with these facts in hand.

### Setup

RL Zoo minigrid-defaults, seed 0, 1M steps per task; periodic evaluation every 10k steps with 20
deterministic (greedy) episodes; final evaluation on 100 held-out episodes.

### Results

Training-episode success (stochastic policy):

| Steps | SimpleCrossing | LavaCrossing |
|---|---|---|
| 0–100k | 0.11 | 0.00 |
| 100–200k | 0.26 | 0.00 |
| 200–400k | 0.31 | 0.00 |
| 400–600k | 0.40 | 0.10 |
| 600–800k | 0.52 | 0.09 |
| 800k–1M | 0.84 | 0.13 |

Final model, 100 held-out episodes:

| Task | Evaluation | Success | Lava death | Timeout |
|---|---|---|---|---|
| SimpleCrossing | greedy | 0.72 | 0.00 | 0.28 |
| SimpleCrossing | stochastic | 0.90 | 0.00 | 0.10 |
| LavaCrossing | greedy | 0.02 | 0.08 | 0.90 |
| LavaCrossing | stochastic | 0.07 | 0.36 | 0.57 |

### What the pilot showed

- Both tasks show the sparse-reward problem; LavaCrossing strongly (no success at all before ~400k steps).
- **Greedy evaluation understates the policy.** The greedy policy mostly fails by timing out: on
  SimpleCrossing greedy evaluation success stayed at 0 until 300k steps while training success was already
  0.11–0.31. Interpretation (one seed each, not a result): the memoryless policy loops when it acts greedily,
  and sampled actions break the loops.
- 1M steps is too short: SimpleCrossing was still rising steeply, and LavaCrossing reached only 0.13.

## EXP-011: baseline

### Design (pre-registered before launch)

- Tasks: both; seeds 0–4 (5 seeds); 3M steps per run.
- Agent: RL Zoo minigrid-defaults, unchanged from the pilot.
- Evaluation: every 20k steps, 20 episodes on fixed held-out seeds 10000–10019 with the **stochastic**
  policy; final evaluation on 100 held-out episodes (seeds 20000–20099), stochastic and greedy.
- Changes from the pilot, each with its reason: 1M → 3M steps (pilot still rising); deterministic →
  stochastic periodic evaluation (greedy loops hide learning progress).
- Primary metrics: final stochastic success, area under the success curve (AUC), steps to 50% and 80%
  success. Secondary: greedy success, death and timeout rates, successful-episode length.
- Decision rules fixed in advance:
  - no hyperparameter changes after seeing results; a crashed run is rerun with the same seed;
  - if LavaCrossing stays below 50% final success in most seeds, it is reported as "not solved by
    sparse-reward PPO in 3M steps";
  - if SimpleCrossing reaches ≥ 95% in all seeds well before 3M, shaping on it is compared by sample
    efficiency (AUC, steps to 80%), not by final success.
- **Addendum, written before the final results were analyzed:** loop metrics (greedy loop rate, revisit
  ratio) were added as secondary, descriptive metrics, because the pilot showed looping. Disclosure: at
  that point the interim SimpleCrossing evaluation rows had been seen while monitoring; the LavaCrossing
  results had not.

### Results

Mean ± SD over 5 seeds:

| Metric | SimpleCrossing | LavaCrossing |
|---|---|---|
| Final success, stochastic (100 episodes) | 1.000 ± 0.000 | 0.760 ± 0.286 (per seed: 0.98, 0.83, 0.30, 1.00, 0.69) |
| AUC of evaluation success, 0–3M | 0.932 ± 0.089 | 0.523 ± 0.316 |
| Steps to 50% success | 0.19 ± 0.26M (5/5 seeds) | 0.83 ± 0.71M (4/5 seeds) |
| Steps to 80% success | 0.26 ± 0.29M (5/5 seeds) | 1.05 ± 0.75M (4/5 seeds) |
| Greedy final success | 0.974 ± 0.048 | 0.620 ± 0.374 |
| Death / timeout rate, stochastic | 0 / 0 | 0.018 / 0.222 |
| Death / timeout rate, greedy | 0 / 0.026 | 0 / 0.380 |

Training-episode outcomes over time (success / death / timeout, mean over 5 seeds; descriptive):

| Steps | SimpleCrossing | LavaCrossing |
|---|---|---|
| 0–50k | 0.25 / 0 / 0.75 | 0.04 / **0.61** / 0.36 |
| 50–100k | 0.54 / 0 / 0.46 | 0.08 / 0.34 / 0.58 |
| 100–250k | 0.77 / 0 / 0.23 | 0.17 / 0.14 / **0.68** |
| 250–500k | 0.82 / 0 / 0.18 | 0.29 / 0.12 / 0.60 |
| 0.5–1M | 0.98 / 0 / 0.02 | 0.48 / 0.09 / 0.43 |
| 1–2M | 1.00 / 0 / 0 | 0.58 / 0.06 / 0.36 |
| 2–3M | 1.00 / 0 / 0 | 0.75 / 0.02 / 0.23 |

Figure: `results/figures/exp011_success.png`. Table: `results/tables/exp011_runs.csv`.

### Observations

1. SimpleCrossing is solved by all seeds (100% final success, ~16-step successful episodes).
2. LavaCrossing is solved by most seeds, but slowly and unreliably: final success 0.30–1.00, steps to 50%
   0.20M–1.84M, and seed 2 never reaches 50% evaluation success.
3. On LavaCrossing, lava deaths dominate only at the very start (61% of training episodes in the first 50k
   steps). After ~100k steps most failures are timeouts.
4. All greedy failures are timeouts, in both tasks; there are no greedy deaths.

### Interpretations

- The agent stops dying early, although the original reward does not punish death (death and timeout both
  give 0). After ~100k steps the obstacle on LavaCrossing is not dying but failing to find and cross the gap
  in 324 steps.
- This only partly supports the expectation that the agent is mostly terminated by lava before it can
  explore the map: that holds only in the first ~100k steps.
- For shaping, this suggests that a goal-directed potential targets the main failure (timeouts), while a
  lava-proximity potential targets deaths, which are already rare after ~250k steps.

### Decisions and reasons

- **The baseline is accepted unchanged as the reference** (pre-registered rule).
- **LavaCrossing is reported as "solved by 4 of 5 seeds within 3M steps, with a large spread across seeds"**,
  because the "not solved" rule (most seeds below 50%) is not met.
- **SimpleCrossing will be compared by sample efficiency** (AUC, steps to 80%), because it meets the ≥ 95%
  rule.
- **10 seeds are planned for the main comparison**: the large LavaCrossing spread (SD 0.29) supports it.
- **Next: behavior analysis**, to answer whether the timeouts come from looping or from not finding the gap.

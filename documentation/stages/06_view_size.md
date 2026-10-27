# Stage 6 — One environmental factor: 5×5 instead of 7×7 view

**Workflow steps:** 4 (change one environmental factor) and 5 (controlled comparison). **Experiment:**
EXP-012. **Status:** done.

## Question

How does shrinking the egocentric view from 7×7 to 5×5 change the sample efficiency and behavior of
sparse-reward PPO on the two tasks?

## Why this factor

- The study is about a limited egocentric view (7×7 by default in MiniGrid). Its size is therefore the
  natural environmental factor to vary.
- It gives the 5×5 sparse-reward reference that the planned experiment "5×5 view × shaping" (EXP-021) needs.
- Running this step after the shaping experiments, or only as a 1-seed pilot, was considered. The first would
  skip workflow step 4 and the second would not allow a comparison, so the step was run with the full 5
  seeds.

## Design (pre-registered before launch)

- Identical to EXP-011 except `env_kwargs: {agent_view_size: 5}`. The flattened observation shrinks from
  2835 to 2763 entries; everything else (agent, seeds 0–4, 3M steps, evaluation, metrics) is unchanged.
- Decision rules: no hyperparameter changes; if a task has 0% final success in all 5 seeds, the EXP-021 design
  for that task is revisited before any shaping run with a 5×5 view.

## Results

Mean ± SD over 5 seeds; 7×7 values are the EXP-011 baseline.

| Metric | Simple 7×7 | Simple 5×5 | Lava 7×7 | Lava 5×5 |
|---|---|---|---|---|
| Final success, stochastic | 1.000 ± 0.000 | 0.966 ± 0.071 | 0.760 ± 0.286 | 0.894 ± 0.068 |
| AUC of evaluation success | 0.932 ± 0.089 | 0.931 ± 0.066 | 0.523 ± 0.316 | 0.504 ± 0.228 |
| Steps to 50% (seeds that reached it) | 0.19 ± 0.26M (5/5) | 0.10 ± 0.10M (5/5) | 0.83 ± 0.71M (4/5) | 1.33 ± 0.81M (5/5) |
| Steps to 80% (seeds that reached it) | 0.26 ± 0.29M (5/5) | 0.16 ± 0.12M (5/5) | 1.05 ± 0.75M (4/5) | 1.58 ± 0.84M (5/5) |
| Greedy final success | 0.974 ± 0.048 | 0.860 ± 0.163 | 0.620 ± 0.374 | 0.756 ± 0.099 |
| Death / timeout, stochastic | 0 / 0 | 0 / 0.034 | 0.018 / 0.222 | 0 / 0.106 |
| Successful-episode length, stochastic | 15.8 ± 0.6 | **26.5 ± 4.1** | 31.7 ± 20.0 | 32.0 ± 9.9 |
| Successful-episode length, greedy | 15.5 ± 0.6 | **19.7 ± 1.0** | 17.0 ± 2.6 | **21.4 ± 2.8** |

Figure: `results/figures/exp011_vs_exp012_success.png`. Table: `results/tables/exp011_vs_exp012_runs.csv`.

## Controlled comparison (exploratory statistics)

Exact two-sided permutation test on the difference of means, 5 vs 5 seeds (`src/permutation_test.py`). With
5 vs 5 seeds the smallest possible p is 2/252 = 0.008. These 16 tests were not pre-registered and are not
corrected for multiple comparisons, so they only indicate directions.

- **Successful-episode length is the only clear difference.** SimpleCrossing: +10.7 steps stochastic
  (p = 0.008) and +4.2 greedy (p = 0.008); all five 5×5 runs are above all five 7×7 runs. LavaCrossing greedy:
  +4.4 steps (p = 0.03).
- **No detectable difference** in final success, AUC, greedy success or steps to 50%/80% on either task (all
  p ≥ 0.11).
- LavaCrossing training success in the first 500k steps: 0.07 (5×5) vs 0.23 (7×7), p = 0.27.

## Observations

1. Both tasks are still solved with a 5×5 view; all 10 runs end at ≥ 82% final success.
2. With 5 seeds, the smaller view causes no measurable change in final success, AUC or steps to 50%/80%.
3. Successful paths are longer: about 4 more greedy steps on both tasks.
4. LavaCrossing 5×5 splits into two groups: seeds 1–3 reach 50% at 0.60–0.86M, seeds 0 and 4 only at ~2.2M.
   In the 7×7 baseline the spread came from a single seed.
5. On LavaCrossing 5×5, timeouts dominate earlier: 79–84% of training episodes between 50k and 500k steps,
   against 58–68% with 7×7.
6. The mean final success on LavaCrossing is higher with 5×5 (0.89 vs 0.76), but this difference is not
   detected with 5 seeds; the 7×7 mean is pulled down mostly by seed 2 (0.30).

## Interpretations

- On 9×9 maps, a 5×5 view mainly changes *how well* the agent navigates, not *whether* it learns to. A
  plausible reason is that it sees the gap and the goal later and has to search more.
- The original reward barely punishes a longer path: four extra steps cost 0.9 · 4 / 324 ≈ 0.011 of the goal
  reward, so sparse-reward PPO has little pressure to shorten paths. A distance potential gives a signal at
  every step and may shorten them.
- Because the 5×5 baseline is not worse in final success, an extra benefit of shaping with a 5×5 view
  (H-view) can only show up in learning speed, path length and the early phase, not in final success.
- The maps are small, so a 5×5 view already covers much of them. Whether the view effect grows on larger maps
  is not tested; this is a limitation, not a planned experiment.

## Decisions and reasons

- **The EXP-012 rule is not triggered** (no task at 0% success), so EXP-021 stays as planned with EXP-012 as
  its 5×5 reference.
- **To consider in the EXP-021 pre-registration, before any shaping result:** reporting successful-episode
  length (stochastic and greedy) and the training outcomes of the first 500k steps next to the primary
  metrics. Path length was the only clear difference between the views; the early-phase difference was not
  detected (p = 0.27). The primary metrics stay those of EXP-011.
- **The main shaping comparison keeps the planned statistics**: 10 seeds, a 95% bootstrap confidence
  interval and the probability of improvement. With 5 vs 5 seeds a permutation test can never go below
  p = 0.008, so 5-seed tests stay exploratory.

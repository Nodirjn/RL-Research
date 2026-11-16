# Stage 12 — EXP-022: the shifted BFS potential with a 5×5 view (pre-registration)

**Workflow step:** 11 (validation and ablations). **Status:** run and analyzed (10 of 10 runs finished; results below).
**Code version:** commit `5fb543e` (the config, the launcher and the analysis scripts
used for this experiment). Everything below was fixed before any EXP-022 run.

## Why this experiment

**Observation in EXP-021** (5×5 view, SimpleCrossing, BFS potential Φ = −b/B;
`results/tables/exp021_runs.csv`, `results/tables/exp021_loops.csv`, the runs' `train_monitor.csv`):

- 3 of the 10 BFS runs (seeds 4, 5 and 8) never learn: AUC 0.00 and final success 0.00. The other seven reach
  a final success of 0.94 to 1.00. All 10 runs of the 5×5 sparse baseline learn (final success 0.84 to 1.00),
  and so do all 10 BFS runs with the 7×7 view (EXP-020, 0.95 to 1.00).
- In these three runs nearly every training episode ends at the time limit: about 9,250 episodes in 3M steps
  (324 steps each), of which 4 to 14 reach the goal. The final greedy policy loops in every evaluation episode
  (greedy loop rate 1.0, revisit ratio 0.87 to 0.92).
- In every BFS run, a training episode that ends at the time limit collects more shaping reward (the
  undiscounted sum of F) than one that reaches the goal: 2.31 to 2.77 against 1.19 to 1.25 in the seven runs
  that learn. Even with the original reward added, the total undiscounted reward of a time-out episode is
  higher in those seven runs: 2.51 to 2.77 against 2.11 to 2.14 for an episode that reaches the goal.

**Candidate explanation (H-offset, [research plan](../research_plan.md#4-research-questions-and-hypotheses-draft-workflow-step-9)).**
With Φ ≤ 0, every step that does not end the episode earns a small extra (γ − 1) · Φ(s′) = +0.01 · b/B, so
stalling and looping away from the goal earn a small bonus. In the discounted return that PPO maximizes, the
shaping of any episode sums to γ^T · Φ(s_T) − Φ(s₀), which hardly depends on the path (γ^324 ≈ 0.04), so this
bonus should not change which policy is best. It can still change learning: the critic has to represent Φ,
which depends on the agent's position, and the 5×5 view shows less of it. The shifted potential
Φ = 1 − b/B (offset 1) turns the bonus into a cost of −0.01 · (1 − b/B) per non-terminating step and leaves
the differences of Φ between cells unchanged.

**Expected direction:** fewer runs that never learn and a lower greedy loop rate than with Φ = −b/B.
**Expected cost** (H-offset, part a): with Φ ≥ 0 the step onto the goal earns F = −Φ(s) ≈ −(1 − 1/B), which
weakens the goal signal, so learning may be slower even if the loops disappear.

## Status of the evidence: exploratory

- The task, the potential and the view were chosen after seeing the EXP-021 results, and the comparison group
  is the same set of EXP-021 runs that showed the failure. Such a comparison favors finding a difference, so
  EXP-022 is exploratory: no confirmatory test and no claim of an effect.
- With 10 runs per group, 3 of 10 against 0 of 10 runs that never learn has a two-sided Fisher exact p of
  0.21. The run counts can show a direction only.
- The runs use the seeds of the EXP-021 BFS runs (0–9), so each seed can also be compared with its EXP-021
  counterpart.

## Condition

| Condition | `shaping` section of the config | Task | Seeds | Runs |
|---|---|---|---|---|
| Shifted BFS, 5×5 | `potential: bfs, offset: 1.0` (Φ = 1 − b/B) | SimpleCrossing | 0–9 | 10 |

- **Comparison groups** (existing runs, not rerun): the EXP-021 BFS runs with the 5×5 view (seeds 0–9) and the
  5×5 sparse baseline (EXP-012 seeds 0–4, EXP-021 seeds 5–9).
- **Config:** `configs/exp022_simplecrossing_bfs_shifted.yaml`, identical to
  `configs/exp021_simplecrossing_bfs.yaml` except for the offset (tested in `tests/test_exp022_config.py`).
  Launcher `src/run_exp022.sh`.
- **Not run:** LavaCrossing, and the shifted Euclidean potential planned as the extension of EXP-020. The
  failure was seen with BFS on SimpleCrossing, and this condition tests H-offset where it was seen
  ([decision 49](../decisions.md)).
- **Smoke test** (4096 steps, seed 0): the config trains with the 5×5 observation; for every training episode
  the logged reward equals the original plus the shaping return; an episode that ends at the time limit has a
  negative shaping return. This is a code check, not a result.

## What is reported

For the shifted condition against each comparison group (on SimpleCrossing, 5×5 view), per group and per
seed, with 95% bootstrap intervals and the probability of improvement where a comparison is made
(`compare_conditions.py --pairs`, no test):

1. Runs that never learn: final success (stochastic policy, 100 held-out episodes) below 0.1.
2. AUC, final success, steps to 50% and 80% success.
3. Greedy loop rate and revisit ratio.
4. The shaping return of training episodes by outcome (`shaping_success`, `shaping_timeout`).

**Rules.**

- No hyperparameter or code change after any EXP-022 result. A crashed run is rerun with the same seed and
  reported.
- Every run is reported; none is excluded.
- During the runs only their status (finished or crashed) is checked.
- Analysis, all at the code commit named at the top:

  ```
  python src/analyze_minigrid.py experiments/EXP-012 experiments/EXP-021 experiments/EXP-022 --name exp022
  python src/loop_metrics.py experiments/EXP-012 experiments/EXP-021 experiments/EXP-022 --name exp022
  python src/behavior.py experiments/EXP-012 experiments/EXP-021 experiments/EXP-022 --name exp022
  python src/compare_conditions.py results/tables/exp022_runs.csv --pairs view5_bfs:view5_bfs_shifted view5_sparse:view5_bfs_shifted
  python src/compare_conditions.py results/tables/exp022_loops.csv --pairs view5_bfs:view5_bfs_shifted view5_sparse:view5_bfs_shifted --metrics greedy_loop_rate revisit_ratio
  ```

## Results (exploratory)

All 10 runs finished; none was rerun or excluded. The analysis is the one listed above, at the code commit
named at the top. Tables: `results/tables/exp022_*`; figure: `results/figures/exp022_success.png`.

| SimpleCrossing, 5×5 view | Sparse | BFS, Φ = −b/B | Shifted BFS, Φ = 1 − b/B |
|---|---|---|---|
| Runs that never learn (final success < 0.1) | 0 of 10 | 3 of 10 (seeds 4, 5, 8) | 2 of 10 (seeds 4, 6) |
| AUC, mean ± SD | 0.937 ± 0.047 | 0.651 ± 0.449 | 0.614 ± 0.370 |
| Greedy loop rate, mean | 0.128 | 0.412 | 0.467 |
| Runs with training success below 0.05 at 500k steps | 0 of 10 | 3 of 10 | 5 of 10 |

- Shifted against plain BFS: AUC −0.037 [−0.376, +0.309], P(B > A) 0.34; greedy loop rate +0.055
  [−0.272, +0.374]; revisit ratio −0.043 [−0.341, +0.259].
- Shifted against sparse: AUC −0.323 [−0.551, −0.119], P(B > A) 0.05; greedy loop rate +0.339 [+0.125, +0.573].
- Seed 4 never learns with either potential; seeds 5 and 8 learn with the offset (seed 5 late, AUC 0.46);
  seed 6 learns with the plain potential (AUC 0.98) but not with the offset.
- The offset reverses the order of the shaping return by outcome, as intended: in the 8 runs that learn, a
  training episode that ends at the time limit collects −0.61 to −0.85 (undiscounted sum of F), one that
  reaches the goal −0.10 to −0.16.
- **Reading.** H-offset is not supported: removing the per-step bonus of a negative potential neither removes
  the runs that never learn nor lowers the greedy loop rate, and early learning is slower (the expected cost).
  The cause of the failure of the BFS potential with the 5×5 view remains open. With 10 runs per group the
  counts show a direction only.

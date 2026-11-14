# Stage 10 — EXP-021: the shaping conditions with a 5×5 view (pre-registration)

**Workflow step:** 11 (validation). **Status:** pre-registered, not started.
**Code version:** commit `(filled in at the code commit)` (the configs, the launcher and the analysis scripts
used for this experiment). Everything below was fixed before any EXP-021 run.

## Questions

- **RQ4.** Is the benefit of shaping larger when the agent sees less (5×5 view)?
- **RQ1 and RQ2 with the 5×5 view.** Do the Euclidean and the BFS potential make PPO learn faster than the
  sparse baseline, and does BFS beat Euclidean, when the view is 5×5?

The hypothesis is H-view of the
[research plan](../research_plan.md#4-research-questions-and-hypotheses-draft-workflow-step-9): PBRS acts
like adding Φ to the critic's value estimate, and Φ uses the global position, which the agent cannot see,
so shaping may help more when the agent sees less. The symbols are defined in the plan's
[notation](../research_plan.md#52-notation).

## Why the RQ4 test uses the walk after crossing, not AUC

**Observations from earlier stages:**

- With the 5×5 view and the sparse reward (EXP-012, 5 seeds), the area under the success curve did not change
  detectably ([stage 6](06_view_size.md)): LavaCrossing 0.50 ± 0.23 against 0.52 ± 0.32 with 7×7,
  SimpleCrossing 0.93 against 0.93.
- What the 5×5 view did change is the walk from the obstacle line to the goal. In successful episodes
  (stochastic policy, final evaluation), the agent needed these steps from its first step on the goal side
  of the line to the goal (`results/tables/exp011_exp012_crossing.csv`):

  | Task | 7×7 view | 5×5 view |
  |---|---|---|
  | SimpleCrossing | 6.2 ± 0.1 | 11.2 ± 2.8 |
  | LavaCrossing | 11.0 ± 3.7 | 13.5 ± 2.3 |

- With the 5×5 view, about 40% of the LavaCrossing timeouts happen after crossing
  ([stage 5](05_behavior_analysis.md)): the goal is often out of a 5×5 view after crossing, while with 7×7
  it is often visible.
- With the 7×7 view, the BFS potential reaches an AUC of 0.84 ± 0.09 on LavaCrossing (EXP-020,
  [stage 9](09_shaping_results.md)); the highest LavaCrossing AUC of any run so far is 0.96. On
  SimpleCrossing the sparse baseline already has an AUC of 0.89.

**Consequences for the design:**

- The interaction on AUC has little room. If the 5×5 sparse baseline has about the AUC of the 7×7 one, as in
  EXP-012, the interaction for BFS on LavaCrossing can be at most about 0.96 − 0.84 ≈ 0.12. With 10 runs per
  group and the spread seen so far, its 95% interval is expected to be about ±0.25 wide or wider. Twenty runs
  per group would narrow it only to about ±0.2. So AUC cannot answer RQ4 at an affordable budget, and its interaction is
  reported as an estimate without a claim.
- The walk after crossing is where a smaller view costs the agent something, and both potentials point to the
  goal there. Behind the line there is no obstacle, so the Euclidean and the BFS potential both increase
  toward the goal. If Φ supplies the missing global information (H-view), shaping should shorten this walk,
  and more so with the 5×5 view.
- **Prediction (H-view):** the interaction contrast I below is negative for both potentials on both tasks.

**Order of the choices:** the metric was fixed in the analysis code (`src/view_interaction.py`) before its
values for the EXP-020 conditions had been computed. They were first printed during a code check of that
script, after the choice, and are not used for any decision here.

## Conditions

| Condition | `shaping` section of the config | Tasks | Seeds | New runs |
|---|---|---|---|---|
| Sparse baseline, 5×5 | none for seeds 0–4 (the EXP-012 runs); `potential: none` for seeds 5–9 | both | 0–9 | 10 |
| Euclidean, 5×5 | `potential: euclidean` (Φ = −d/D) | both | 0–9 | 20 |
| BFS (geodesic), 5×5 | `potential: bfs` (Φ = −b/B) | both | 0–9 | 20 |

- The 7×7 groups of the interaction are those of EXP-020: the sparse baseline (EXP-011 seeds 0–4 and EXP-020
  seeds 5–9), Euclidean and BFS.
- Everything else is identical to EXP-012, and to EXP-020 apart from the view: RL Zoo MiniGrid defaults,
  3M steps, evaluation every 20k steps on 20 held-out episodes (seeds 10000–10019, stochastic policy), final
  evaluation on 100 held-out episodes (seeds 20000–20099, stochastic and greedy), checkpoints at 1M and 2M.
- The naive control and the lava term are not repeated with the 5×5 view. RQ3 and the lava question are
  answered with 7×7 (stage 9), and neither bears on RQ4.
- **Order:** LavaCrossing, then SimpleCrossing (50 runs). Within each task the runs are ordered seed by seed
  across conditions, so that an early stop leaves every condition with about the same number of seeds.
- **Files:** `configs/exp021_<task>_<condition>.yaml` (the EXP-012 configs with the `shaping` section of
  EXP-020); launcher `src/run_exp021.sh`.
- **Smoke test** (4096 steps, one seed per config): every config trains with the 5×5 observation (the same
  size as the EXP-012 models); for every training episode the logged reward equals the original plus the
  shaping return; `potential: none` adds no shaping; the analysis chain below reads the runs. The smoke
  policies have no successful episodes, so the RQ4 script was also run on tables built from the EXP-012 runs.
  This is a code check, not a result.
- **Why the 5×5 baseline can be pooled:** `potential: none` leaves the reward unchanged and only adds the
  outcome columns to the training log (tested). The current code reproduces the EXP-012 evaluations episode
  by episode ([stage 7](07_shaping_method.md#reproducibility-of-the-earlier-runs)), and re-running the loop
  metrics of EXP-012 with the current code gives byte-identical files.

## Metrics

All metrics of [stage 8](08_shaping_experiment.md#metrics), and three behavior metrics computed by
`behavior.py` from the final-evaluation episodes:

- **Steps after crossing:** steps from the first step on the goal side of the line to the goal, mean over the
  successful episodes of a run (stochastic policy; the greedy version is exploratory). A run without a
  successful episode has no value and is left out; the number of runs per group is reported.
- **Steps to cross:** steps until the first step on the goal side, mean over the successful episodes.
- **After-crossing timeouts:** timed-out episodes in which the agent had crossed the line, as a fraction of
  all 100 episodes.

## Statistics and decision rules

**Confirmatory, RQ1 and RQ2 with the 5×5 view.** One metric, AUC, and six comparisons, all with the 5×5 view:

| # | Condition B against condition A | Tasks |
|---|---|---|
| 1 | Euclidean against sparse | both |
| 2 | BFS against sparse | both |
| 3 | BFS against Euclidean | both |

Tested as in [stage 8](08_shaping_experiment.md#statistics-and-decision-rules): the exact two-sided
permutation test of the difference of mean AUC (10 against 10 runs), Holm-corrected over the six at a
family-wise level of 0.05; effect sizes are the difference of means with a 95% bootstrap interval and the
probability of improvement.

**Confirmatory, RQ4.** For each task and potential S (Euclidean, BFS), the interaction contrast

```
I = (S − sparse, 5×5 view) − (S − sparse, 7×7 view)
```

where each term is a difference of group means of the steps after crossing. Four contrasts (2 tasks ×
2 potentials).

- Interval: percentile bootstrap, 10,000 resamples of the runs within each of the four groups, at the level
  1 − 0.05/4 = 98.75% (Bonferroni over the four contrasts).
- Decision: "shaping shortens the walk to the goal more with the 5×5 view" is claimed for a contrast only if
  its whole interval lies below 0. If it lies above 0, the opposite is reported. Otherwise the result is
  "no detectable difference".
- Why an interval and not a permutation test: an interaction of two factors has no exact permutation test,
  because the main effects of shaping and of the view must stay in the data. The bootstrap is the method
  already used for all effect sizes. With 10 runs per group, percentile intervals can be somewhat too narrow,
  so the decision uses the more conservative 98.75% level.
- Caveat: the metric is averaged over successful episodes. A condition that also succeeds on harder maps can
  have longer walks, so final success is reported next to it. On SimpleCrossing the sparse, Euclidean and BFS
  conditions run so far have a final success of about 0.97 or more, so this matters mainly on LavaCrossing.

The two confirmatory families answer different questions and are each held at a family-wise level of 0.05.

**Estimates without a claim.** The interaction on AUC, final success and steps to 50%/80%, with 95% intervals.

**Exploratory.** The interaction on the greedy steps after crossing, the steps to cross, the after-crossing
timeouts, the revisit ratio and the greedy loop rate, with 95% intervals; and the EXP-020 hypothesis checks
that have both conditions with the 5×5 view:

| Hypothesis | Metric | Comparison (5×5 view) |
|---|---|---|
| H-dead-end | Never-crossed timeouts, revisit ratio | BFS against Euclidean |
| H-lava-a | Training death rate in the first 500k steps | Euclidean against sparse (LavaCrossing) |

**Rules.**

- No hyperparameter or code change after any EXP-021 result. A crashed run is rerun with the same seed and
  reported.
- Every run is reported; none is excluded.
- During the runs only their status (finished or crashed) is checked. Evaluation results are analyzed after
  all runs have finished.
- Analysis, all at the code commit named at the top:

  ```
  python src/analyze_minigrid.py experiments/EXP-012 experiments/EXP-021 --name exp021
  python src/loop_metrics.py experiments/EXP-012 experiments/EXP-021 --name exp021
  python src/behavior.py experiments/EXP-012 experiments/EXP-021 --name exp021
  python src/compare_conditions.py results/tables/exp021_runs.csv --family exp021
  python src/hypothesis_checks.py --name exp021
  python src/view_interaction.py --view7 exp020 --view5 exp021
  ```

  The 7×7 tables (`exp020_*`) are those of stage 9 and are not recomputed.

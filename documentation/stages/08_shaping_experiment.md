# Stage 8 — EXP-020: potential-based shaping against the sparse baseline (pre-registration)

**Workflow steps:** 10–11 (method; validation and ablations). **Status:** pre-registered, not started.
**Code version:** commit `4c6b0c6` (the configs, the launcher and the analysis scripts used for this
experiment). Everything below was fixed before any shaping run.

## Questions

- **RQ1.** Does PBRS with a distance potential make PPO learn faster on SimpleCrossing and LavaCrossing
  (7×7 view), without lowering final success?
- **RQ2.** Does the shape of the potential matter: Euclidean against path-length (BFS, geodesic) distance, and
  with or without a lava-proximity term?
- **RQ3.** Does the potential form matter: PBRS against a naive progress bonus that is not potential-based?

RQ4 (view size) is answered by EXP-021, which gets its own pre-registration. The hypotheses are those of the
[research plan](../research_plan.md#4-research-questions-and-hypotheses-draft-workflow-step-9); the symbols
are defined in its [notation](../research_plan.md#52-notation).

## Conditions

| Condition | `shaping` section of the config | Tasks | Seeds | Phase |
|---|---|---|---|---|
| Sparse baseline | none for seeds 0–4 (the EXP-011 runs); `potential: none` for seeds 5–9 | both | 0–9 | core |
| Euclidean | `potential: euclidean` (Φ = −d/D) | both | 0–9 | core |
| BFS (geodesic) | `potential: bfs` (Φ = −b/B) | both | 0–9 | core |
| Naive control | `potential: euclidean, naive: true` | both | 0–9 | core |
| Euclidean + lava | `potential: euclidean, lava_penalty: 0.11785` (c = 1/D) | LavaCrossing | 0–9 | core |
| Shifted | `potential: euclidean, offset: 1.0` (Φ = 1 − d/D) | LavaCrossing | 0–9 | extension |

- Everything else is identical to EXP-011: RL Zoo MiniGrid defaults, 7×7 view, 3M steps, evaluation every
  20k steps on 20 held-out episodes (seeds 10000–10019, stochastic policy), final evaluation on 100 held-out
  episodes (seeds 20000–20099, stochastic and greedy), checkpoints at 1M and 2M.
- **Order:** the LavaCrossing core runs, then the SimpleCrossing core runs (80 runs), then EXP-021, then the
  extension (10 runs). Within each part the runs are ordered seed by seed across conditions, so that an early
  stop leaves every condition with about the same number of seeds.
- **Files:** `configs/exp020_<task>_<condition>.yaml` (the EXP-011 configs with a `shaping` section);
  launcher `src/run_exp020.sh core`, then `src/run_exp020.sh extension`.
- **Smoke test** (4096 steps, one seed per config): every config trains; for every training episode the
  logged reward equals the original plus the shaping return; `potential: none` adds no shaping; the
  analysis scripts read the runs. This is a code check, not a result.
- **Why the baseline can be pooled:** `potential: none` leaves the reward unchanged and only adds the outcome
  columns to the training log (tested). The current code reproduces the EXP-011 evaluations episode by
  episode ([stage 7](07_shaping_method.md#reproducibility-of-the-earlier-runs)), so seeds 0–4 of EXP-011 and
  seeds 5–9 of this experiment form one 10-seed baseline.

## Metrics

As in the [research plan](../research_plan.md#7-metrics), computed per run:

- **Primary:** area under the success curve (AUC), final success rate (stochastic policy), steps to 50% and
  80% success.
- **Secondary:** greedy final success; death and timeout rates; successful-episode length; greedy loop rate;
  revisit ratio; never-crossed timeouts; success, death and timeout rates of the training episodes in the
  first 500k steps; shaping return by outcome.

## Statistics and decision rules

**Confirmatory comparisons.** One metric, AUC, and nine pre-specified comparisons:

| # | Condition B against condition A | Question | Tasks |
|---|---|---|---|
| 1 | Euclidean against sparse | RQ1 | both |
| 2 | BFS against sparse | RQ1 | both |
| 3 | BFS against Euclidean | RQ2, H-dead-end | both |
| 4 | Euclidean + lava against Euclidean | RQ2, H-lava-b | LavaCrossing |
| 5 | Naive control against Euclidean | RQ3, H-naive | both |

- Test: the exact two-sided permutation test of the difference of mean AUC (10 against 10 runs, all 184,756
  splits). The nine p-values are corrected together with the Holm method, at a family-wise level of 0.05.
  A difference is claimed only if its corrected p is below 0.05.
- Effect size: the difference of means with a 95% bootstrap confidence interval (10,000 resamples of the runs
  within each condition) and the probability of improvement, P(B > A) over all run pairs (Agarwal et al.,
  2021).
- **Why AUC:** RQ1 is about learning speed, and on SimpleCrossing the baseline already reaches a final
  success of 1.00 ([decision 15](../decisions.md)), so only a sample-efficiency metric can differ on both
  tasks.

**Other primary metrics.** Final success and steps to 50%/80% are reported for the same pairs with the
difference of means, its 95% bootstrap interval and the probability of improvement, without a test. For
steps to a threshold, a run that never reaches it counts as 3M. "Without lowering final success" (RQ1) holds
unless the interval of the final-success difference lies entirely below 0.

**Hypothesis checks (exploratory).** Reported with intervals; a result is described as "consistent with" a
hypothesis, never as confirming it:

| Hypothesis | Metric | Comparison |
|---|---|---|
| H-dead-end | Never-crossed timeouts, revisit ratio | BFS against Euclidean |
| H-lava-a | Training death rate in the first 500k steps | Euclidean against sparse (LavaCrossing) |
| H-lava-b | Never-crossed timeouts | Euclidean + lava against Euclidean |
| H-naive | Greedy loop rate, revisit ratio, shaping return by outcome | Naive control against Euclidean |
| H-offset | Training death and success rates in the first 500k steps, step of the first success | Shifted against Euclidean (extension) |

**Rules.**

- No hyperparameter or code change after any shaping result. A crashed run is rerun with the same seed and
  reported.
- Every run is reported; none is excluded.
- During the runs only their status (finished or crashed) is checked. Evaluation results are analyzed after
  all core runs have finished.
- A condition that never reaches 50% success is reported as it is.
- Analysis: `analyze_minigrid.py`, `loop_metrics.py` and `behavior.py` (as for the baseline) and
  `compare_conditions.py` (the statistics above), all at the code commit named at the top.

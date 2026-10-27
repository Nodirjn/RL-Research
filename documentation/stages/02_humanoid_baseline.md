# Stage 2 — Preliminary baseline on MuJoCo Humanoid-v5 (archived)

**Workflow step:** 2 (baseline), first attempt. **Experiments:** EXP-001, EXP-002. **Status:** done, then
archived when the study moved to MiniGrid ([stage 3](03_switch_to_minigrid.md)). Files:
`archive/humanoid/`.

## Question

Does the PPO pipeline reproduce published PPO performance on Humanoid, which standard configuration
should be the baseline, and what return and behavior does that baseline reach?

## Why this stage

Before any change to the reward can be judged, the pipeline itself must be shown to be correct: a baseline
that underperforms because of a bug would make any "improvement" meaningless. The check is made against
published numbers, and the configuration is chosen by a rule fixed before the results.

## What was done

### Reference values collected first

- RL Zoo has a tuned PPO configuration for Humanoid-v4 but no PPO benchmark row for Humanoid.
- Published PPO values: CleanRL (Humanoid-v4, 1M steps, 1 seed): 378.9 ± 50.1; Gao et al. (arXiv 2603.10199,
  preprint; Humanoid-v5, 10 seeds): 671.1 ± 128.3 at 1M and 941.5 ± 288.5 at 3M; SLM-Lab (Humanoid-v5, 10M):
  2661 (a moving average with a different implementation, indicative only).
- Gao et al. report practically identical PPO results on v4 and v5 (669 vs 671 at 1M), so Humanoid-v5 was
  used for everything, and v4 references were kept as secondary reference points.

### EXP-001: benchmark check and configuration selection

Three configurations, all with the original reward:

- **C1 `hist`:** the earlier configuration (4 environments, n_steps 2048, batch 64, lr 3e-4, no normalization).
- **C2 `cleanrl`:** SB3 matched to CleanRL (1 environment, n_steps 2048, lr 3e-4 annealed, observation and
  reward normalization).
- **C3 `zoo`:** the RL Zoo tuned Humanoid-v4 configuration, unchanged.

Pre-registered rules: the pipeline is accepted if C2 at 1M lies in the range of published 1M values
(~380–930); the baseline configuration is the one with the highest mean final return at 3M.

Before the runs, a smoke test confirmed that the same seed gives bitwise-identical evaluations, training
episodes, policy weights and normalization statistics, and that a reloaded model reproduces its evaluation
exactly.

| Stage | Result |
|---|---|
| a: C2, seeds 0–2, 1M | 620.3 ± 115.0 (last 5 evaluations) → inside ~380–930: **pipeline accepted** |
| b: C1/C2/C3, seeds 0–2, 3M, final return (50 held-out episodes) | C1 530.1 ± 67.6, C2 636.2 ± 15.6, **C3 996.0 ± 125.2** → **C3 selected** |

C3 at 3M (last 5 evaluations 954.2 ± 95.5) matches Gao et al. on v5 (941.5 ± 288.5). Every C3 seed was above
every C1 and C2 seed. All 450 held-out episodes ended in a fall.

### EXP-002: full baseline

C3, seeds 0–4, 10M steps. The budget was 10M because the C3 curve was still rising at 3M, 10M is RL Zoo's own
budget for this configuration, and the SLM-Lab reference is at 10M. Fresh 10M runs were used instead of
continuing the 3M runs, so that one command reproduces the baseline; seeds 0–2 matched the EXP-001 runs
exactly up to 3M, which doubled as a determinism check.

| | last 5 at 1M | at 3M | at 10M | final, 50 held-out episodes |
|---|---|---|---|---|
| Mean ± SD over 5 seeds | 670.6 ± 38.4 | 885.0 ± 122.4 | 953.4 ± 78.8 | 972.5 ± 102.5 |

## Observations

- The return plateaus: +214 from 1M to 3M, but only +68 from 3M to 10M.
- **The robot never walks.** All 250 held-out episodes end in a fall; the longest lasted 254 steps (3.8 s),
  and the longest of ~413k training episodes 478 steps (7.2 s). No episode reached the time limit.
- From 1M on, the behavior is the same: a bounding sprint with flight phases (airborne 40–50% of steps),
  mostly right-foot-dominant, then a fall. Training mainly increased speed (1.38 → 2.14 m/s).
- Exploration noise did not collapse (policy std 0.135 → ~0.116 over 10M), so "PPO stopped exploring" is not
  supported as the main reason for the plateau.

## Interpretations

- The tuned baseline converges early to a stable "sprint then fall" strategy. The survival bonus would favor
  staying up much longer, but the agent never discovers how.
- Candidate explanations (none tested): the short discount horizon of this configuration (γ = 0.95), low
  exploration, or an environment property that makes a quick fall unavoidable for this policy class.

## Decisions and reasons

- **C3 × 5 seeds × 10M became the official Humanoid baseline**, by the pre-registered rule.
- **The Humanoid direction was later stopped** (see [stage 3](03_switch_to_minigrid.md)): Humanoid is
  locomotion with full state and an already dense reward, not sparse-reward navigation with a limited view,
  and its baseline did not reach the intended behavior. All files were moved to `archive/humanoid/`; nothing
  was deleted.

## What carried over to MiniGrid

The pipeline: config-driven training, held-out evaluation seeds, pre-registered metrics and decision rules,
determinism checks, and the habit of checking behavior (videos, termination causes), not only the return.

# Stage 7 — Shaping method: implementation and tests (no shaping run yet)

**Workflow step:** 10 (method), prepared ahead of steps 7–9 (see the note in the
[research plan](../research_plan.md#2-workflow)). **Status:** code written and tested. The shaping experiment
(EXP-020) starts after its pre-registration: the core conditions first, the extensions after them.

## Question

How must potential-based shaping be implemented so that it keeps the guarantees of PBRS in this setting
(PPO, episodic tasks with lava deaths and a time limit), and so that its effect can be measured cleanly?

## Why this stage

Small implementation details decide whether shaping really is "potential-based": the discount factor, the
potential of terminal states, the treatment of the time limit, and whether the evaluation ever sees the
shaped reward. These rules were fixed and tested before any shaping run, so that no result can influence
them.

## Implementation rules and their reasons

The symbols are defined in the [notation](../research_plan.md#52-notation) of the research plan. The bonus
added to the reward of every training step is

```
F(s, s′) = γ · Φ(s′) − Φ(s)
```

| Rule | Reason |
|---|---|
| γ = 0.99 in F, the same as PPO's discount | The policy-invariance result uses the discount of the problem being optimized, which here is PPO's γ |
| Φ(s′) = 0 when the episode terminates (goal or lava), so the last step earns F = −Φ(s) | Policy invariance in episodic tasks requires zero terminal potentials (Grześ, 2017) |
| Normal formula on truncation (time limit) | SB3 bootstraps the value at truncation, so the state is not terminal |
| Distance potentials divided by the start's distance (D or B), the same in all conditions: Φ = −1 at the start, 0 at the goal | Scaling does not change the ratio between step bonuses; it fixes the size of shaping relative to the goal reward, and must be the same everywhere |
| Shaping only in the training environment | Every condition is evaluated on the same original reward |
| Outcomes from the environment state | A shaped lava step can have a positive reward |
| VecNormalize identical in all conditions | A fair comparison; its rescaling of the shaped reward is a stated limitation |
| Commit before every launch; hash in the pre-registration | The exact code of every run can be recovered |

## What was built

- **`src/shaping.py`, class `PBRS`:** a wrapper with the potentials `euclidean`, `bfs` and `none`, an optional
  lava penalty c, an optional offset (for Φ = 1 − d/D), and the naive control. At the end of each episode it
  reports the outcome, the original return and the shaping return.
- **`src/train.py`:** wraps the training environment in `PBRS` when the config has a `shaping` section and
  takes γ from the PPO settings, so rule 1 holds by construction. Evaluation still builds its environment
  with `make_env`, without the wrapper. `shaping: {potential: none}` gives exactly the original reward with
  the same log columns; it is meant for the 5 extra baseline seeds.
- **`src/analyze_minigrid.py`:** uses the logged outcome, which every run with a `shaping` section writes.
  Only older, unshaped runs without it fall back to `reward > 0`, which is correct for the original reward.

### Design choices made here

- **BFS potential normalization.** The BFS potential is Φ = −b/B, where B is the path length from the start
  (not the largest path length on the map). Both distance potentials then have Φ = −1 at the start and Φ = 0
  after termination, so the discounted shaping return of every terminated episode (goal or lava) is exactly
  +1 under both (derivation in the [research plan](../research_plan.md#53-shaping)), and the comparison
  between them isolates *how* the signal is spread over the steps. Cells farther from the goal than the
  start get Φ < −1 (in 210 of 500 maps, down to −1.5). The equality holds only for the discounted sum, not
  for truncated episodes, and not for the shifted potential (Φ = 0 at the start).
- **Naive control.** F = max(0, Φ(s′) − Φ(s)), with no γ. At termination it uses the potential of the cell
  the agent actually ends on. With the PBRS convention (Φ = 0 after termination) instead, entering lava would
  earn max(0, 0 − Φ(s)) = +d/D, and the control would test that convention rather than the missing
  potential form.

## Tests

`tests/test_shaping.py` (26 tests in the whole suite, all passing) checks, among others:

- the formula on random steps, the goal step and the lava step with Φ = 0 after termination (the lava step
  gets F = −Φ(s) > 0, as H-lava-a predicts, and is still counted as a death), and the truncation formula;
- with γ = 1, the bonuses of an episode add up to Φ(last state) − Φ(first state), and the original return
  equals that of the unshaped environment;
- with γ = 0.99, the discounted shaping return of a terminated episode is +1 for both distance potentials;
- along the shortest path from the start, the path length b falls by exactly 1 per step (so the BFS potential
  rises by 1/B), the path has exactly B steps, and it goes through the gap;
- the Euclidean dead end: on the start side of a vertical line in column k, the Euclidean potential is
  highest at (k − 1, 7), not at the gap;
- going forward, turning around, going back and turning around again, three times over, ends where it started.
  The naive control pays the one-step progress bonus once per round trip (3 times in total); PBRS with γ = 1
  pays 0;
- evaluation never sees shaping: the evaluation environment contains no PBRS layer, and evaluation returns
  are identical with and without a `shaping` section in the config.

A short smoke test (8k steps, one seed) confirmed that the shaped, `none` and unshaped configs run, that the
logged reward equals original + shaping for every training episode, and that evaluation returns stay in
[0, 1]. It is a code check, not a result.

## Reproducibility of the earlier runs

EXP-010, EXP-011 and EXP-012 ran before the project was under version control, and the shared code changed
afterwards (outcome detection, loop recording). The current code was checked against them: it reproduces
their greedy final evaluations episode by episode (all 2000 episodes of EXP-011/012, and 100 of 100 in each
EXP-010 pilot). From EXP-020 on, every run is launched from committed code.

## What is still open

- The lava penalty size c, the final condition list and the number of seeds are fixed in the EXP-020
  pre-registration.
- The expectations H-lava-a, H-lava-b, H-offset, H-dead-end and H-naive ([research plan](../research_plan.md#4-research-questions-and-hypotheses-draft-workflow-step-9))
  are tested by the pre-registered secondary metrics, in particular the outcomes of the first 500k training
  steps, the loop metrics and the never-crossed fraction.

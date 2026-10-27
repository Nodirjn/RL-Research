# Research plan

**Topic:** potential-based reward shaping (PBRS) to improve the sample efficiency of PPO in sparse-reward
navigation with a limited egocentric view.

**Status:** workflow steps 1–5 are done (audit, baseline, behavior analysis, one environmental change,
controlled comparison). The shaping method is implemented and unit-tested. No shaping experiment has been
run yet; its conditions are fixed in a pre-registration before the first run. What was done in each step,
and why, is in
[`stages/`](stages/); every decision and its reason is in [`decisions.md`](decisions.md).

## 1. Motivation

With a sparse reward the agent is rewarded only when it reaches the goal. Early in training a random policy
reaches the goal rarely, so PPO gets almost no learning signal. Reward shaping adds a dense signal, but an
arbitrary bonus can change what the agent learns: for example, an agent can learn to circle near the goal to
collect a progress bonus. PBRS adds the following bonus to the reward of every step from state s to the
next state s′:

```
F(s, s′) = γ · Φ(s′) − Φ(s)
```

Here Φ(s), the *potential*, is a score of how good it is to be in state s (for example: closer to the goal
means higher), and γ = 0.99 is the discount factor. A step to a state with a higher potential earns a
positive bonus, a step back earns a negative one. Ng, Harada and Russell (1999) showed that a bonus of this
form leaves the optimal policy unchanged. The study measures whether such a signal makes PPO learn faster on
two MiniGrid tasks, and which design choices of Φ matter. All symbols are listed in
[section 5.2](#52-notation).

## 2. Workflow

The study follows an experiment-first workflow: the problem is identified from the agent's behavior before
the literature defines the method.

| Step | What | Status | Where |
|---|---|---|---|
| 1 | Audit of the existing code and results | Done | [01_audit](stages/01_audit.md) |
| 2 | Standard PPO baseline with the original reward | Done (MiniGrid; an earlier Humanoid baseline is archived) | [02](stages/02_humanoid_baseline.md), [04](stages/04_minigrid_baseline.md) |
| 3 | Behavior analysis: how and where the baseline fails | Done for the final policies | [05](stages/05_behavior_analysis.md) |
| 4 | Change one environmental factor | Done: 5×5 instead of 7×7 view | [06](stages/06_view_size.md) |
| 5 | Controlled comparison of the two settings | Done (5 seeds, exploratory statistics) | [06](stages/06_view_size.md) |
| 6 | Problem statement from the evidence | Draft (section 3) | this file |
| 7 | Literature review with verified citations | In progress (section 9) | — |
| 8 | Research gap | Not started | — |
| 9 | Research questions and hypotheses | Draft (section 4) | this file |
| 10 | Method | PBRS implemented and tested; experiment not pre-registered yet | [07](stages/07_shaping_method.md) |
| 11 | Validation and ablations | Planned (section 6) | this file |
| 12 | Paper | Not started | — |

One change of order: the shaping code (step 10) was written before the literature review (step 7). The
reason is practical: the experiments need correct, tested code before they can run, and the PBRS
foundations had to be read to implement it. The code produces no result by itself, and the research gap
(step 8) is still derived after the problem statement.

## 3. Problem statement (draft, workflow step 6)

With the original sparse reward and a 7×7 view, PPO learns SimpleCrossing in every seed, but LavaCrossing
slowly and unreliably: final success ranges from 0.30 to 1.00 across 5 seeds, and one seed never reaches 50%
evaluation success in 3M steps. Lava deaths dominate only at the very start (61% of training episodes in the
first 50k steps). After ~100k steps, the main failure is the time limit: the memoryless agent circles in a
small region, almost always on the start side of the lava line, and never finds the gap (87–88% of timed-out episodes never cross the line). The sparse
reward gives no signal that points toward the gap until the goal is reached.

## 4. Research questions and hypotheses (draft, workflow step 9)

Working research questions (final wording after the literature review):

1. Does PBRS with a distance potential improve the sample efficiency of PPO (area under the success curve,
   steps to 50%/80% success) on SimpleCrossing and LavaCrossing, without lowering final success?
2. Does the shape of the potential matter: straight-line (Euclidean) distance against path-length (BFS)
   distance, and with or without a lava-proximity term?
3. Does the potential form matter: PBRS against a naive progress bonus that is not potential-based?
4. Is the benefit of shaping larger when the agent sees less (5×5 view)?

Hypotheses, stated before any shaping result (each is a prediction to test, not a finding). The symbols
(d, D, b, c, …) are defined in [section 5.2](#52-notation).

- **H-dead-end.** On the start side of the line, the Euclidean potential is highest at the cell next to the
  line that is closest to the goal, not at the gap. For a vertical line in column k, that cell is (k − 1, 7):
  a dead end unless the gap is in the last row. The Euclidean potential can therefore pull the agent away
  from the gap. The BFS potential increases only along paths through the gap and has no such dead end.
- **H-lava-a.** Take the Euclidean potential Φ = −d/D. Because Φ = 0 after the episode terminates, the step
  into lava from a cell at distance d earns

  ```
  F = γ · 0 − Φ(s) = + d/D        (for example +0.5 halfway to the goal)
  ```

  while a step that reduces d by 1 earns only about +0.12. Early in training,
  before the critic has learned that dying ends all reward, lava deaths may therefore increase compared with
  the sparse baseline.
- **H-lava-b.** The lava term lowers Φ by c on every cell next to lava. It cannot penalize the fatal step
  itself: leaving a lava-adjacent cell earns +c from this term, whether the agent steps away from the lava or
  into it (Φ = 0 after termination). It only penalizes approaching lava (about −c). The only gap lies in the
  lava line, so the agent must stand next to lava to cross. Because the main failure is not crossing, this
  term may make crossing harder.
- **H-offset.** With the shifted potential Φ = 1 − d/D, Φ ≥ 0 on every cell, so the last step of a terminated
  episode earns F = −Φ(s) ≤ 0. Dying is now penalized, but so is reaching the goal: from the cell next to the
  goal, F = −(1 − 1/8.485) = −0.882. This is expected to trade fewer early deaths against a weaker goal signal.
- **H-naive.** The naive bonus F = max(0, Φ(s′) − Φ(s)) pays for every step toward the goal but never charges
  a step away. A move forward and back therefore earns a net positive bonus, and repeating it earns more (the
  cycle argument of Ng et al., 1999). It should increase looping, which is already the main failure.
- **H-view.** PPO's advantage estimate (GAE) is built from the TD error. With shaping, this TD error equals
  the unshaped one computed with the critic V replaced by V + Φ:

  ```
  r + F + γ · V(s′) − V(s)  =  r + γ · [V(s′) + Φ(s′)] − [V(s) + Φ(s)]
  ```

  So PBRS acts like adding Φ to the critic's value estimate (exactly so without reward normalization, which
  rescales F). Φ uses the global position, which the agent cannot see, so shaping may help more with a 5×5
  view. The analogous Q-learning result is Wiewiora (2003). This must be compared with asymmetric
  actor-critic methods (Pinto et al., 2018) before any novelty claim.

## 5. Method

### 5.1 Environment and agent (fixed for all conditions)

- **Tasks:** `MiniGrid-SimpleCrossingS9N1-v0` (one wall line with one gap) and
  `MiniGrid-LavaCrossingS9N1-v0` (one lava line with one gap; entering lava ends the episode with reward 0).
  9×9 grid, start at (1, 1) facing right, goal at (7, 7), limit 324 steps. The layout is regenerated on every
  reset, so the agent must learn a general strategy, not one map.
- **Original reward:** r = 1 − 0.9 · (steps taken / 324) on reaching the goal, else 0.
- **Observation:** a 7×7 egocentric partial view (5×5 in the view-size conditions), flattened by
  `FlatObsWrapper`. The policy has no memory.
- **Agent:** Stable-Baselines3 PPO with the RL Zoo MiniGrid defaults, unchanged: 8 environments, n_steps 128,
  batch 64, 10 epochs, learning rate 2.5e-4, γ 0.99, λ 0.95, clip 0.2, entropy coefficient 0, observation and
  reward normalization (VecNormalize). 3M environment steps per run.

### 5.2 Notation

| Symbol | Meaning |
|---|---|
| s, s′ | The state before and after one step |
| p = (x, y) | The agent's cell: x is the column, y the row (1 to 7 inside the outer walls) |
| r | The original reward of the environment |
| Φ(s) | The potential of state s. All potentials here depend only on the agent's cell |
| F | The shaping bonus. During training the agent receives r + F at every step |
| γ | The discount factor, 0.99, the same as PPO's |
| d | Straight-line (Euclidean) distance from the agent's cell to the goal (7, 7): d = √((x − 7)² + (y − 7)²) |
| D | d at the start cell (1, 1): D = √72 ≈ 8.485 |
| b | Length of the shortest path from the agent's cell to the goal that avoids walls and lava, in up/down/left/right moves (found by breadth-first search, BFS) |
| B | b at the start cell; it depends on the map |
| c | The lava penalty (> 0), fixed before the run |
| V(s) | The critic's value estimate of state s |

### 5.3 Shaping

All shaped conditions add the bonus F to the reward of every training step (implemented in
`src/shaping.py`):

```
F(s, s′) = γ · Φ(s′) − Φ(s)
```

Rules fixed before any run:

1. γ = 0.99, the same γ as PPO.
2. When the episode terminates (goal or lava), Φ(s′) = 0, so the last step earns F = −Φ(s). On truncation
   (time limit) the normal formula is used, matching SB3, which bootstraps the value at truncation. Zero
   terminal potentials are needed for policy invariance in episodic tasks (Grześ, 2017).
3. Distance potentials are divided by the start's distance (D or B), so that Φ = −1 at the start and Φ = 0
   at the goal. The scale is the same in all conditions.
4. Shaping is applied only to the training environment. Evaluation always uses the original reward.
   Original and shaped returns are logged in separate columns.
5. Success and death are read from the environment state (the agent's cell), never from the sign of the
   reward. With shaping, a lava death can have a positive reward.
6. Reward normalization (VecNormalize) is identical in all conditions. It rescales the shaped reward; this
   is reported as a limitation.
7. Every run is launched from committed code, and the commit hash is written into the pre-registration.

Potentials:

| Name | Potential Φ | Note |
|---|---|---|
| Euclidean | Φ = −d / D | Core condition: progress toward the goal |
| Euclidean + lava | Φ = −d / D − c on cells next to lava (up, down, left or right); Φ = −d / D elsewhere | Core condition: risk aversion near lava (LavaCrossing only) |
| BFS | Φ = −b / B | Extension. b goes around walls and lava, so it decreases only along paths through the gap |
| Shifted (ablation) | Φ = 1 − d / D | Extension, LavaCrossing only |
| Naive control | Not a potential: F = max(0, Φ(s′) − Φ(s)) with the Euclidean Φ and no γ | Extension. At termination it uses Φ of the cell the agent ends on, not 0 |

**Why both distance potentials give the same total.** Take an episode of T steps that starts in state s₀ and
ends in state s_T, and let F₀, F₁, … be the bonuses of its steps. In the discounted sum, every intermediate
Φ appears once with a plus and once with a minus sign, so only the first and the last state remain:

```
F₀ + γ·F₁ + γ²·F₂ + … + γ^(T−1)·F_(T−1)  =  γ^T · Φ(s_T) − Φ(s₀)
```

If the episode terminates (goal or lava), the last state has Φ = 0 and the start has Φ = −1, so the sum is
exactly +1 under both distance potentials, whatever path the agent takes. They differ only in how this +1 is
spread over the steps. This does not hold for the undiscounted sum, for truncated episodes (the last Φ is not
0), or for the shifted potential (Φ = 0 at the start).

## 6. Experiments

Each experiment changes one thing against the baseline. "Seeds × steps" is per task.

| ID | What changes | Tasks | Seeds × steps | Question | Status |
|---|---|---|---|---|---|
| EXP-010 | Pilot | both | 1 × 1M | Does the pipeline work; which evaluation mode is informative? | Done |
| EXP-011 | Nothing: baseline, original sparse reward, 7×7 view | both | 5 × 3M | How sample-efficient is sparse-reward PPO? | Done |
| EXP-012 | Environmental factor: 5×5 view | both | 5 × 3M | How does a smaller view change learning and behavior? | Done |
| EXP-020 | PBRS, Euclidean potential | both | 10 × 3M | Does a distance signal speed up learning and reduce loops? | Planned (core) |
| EXP-020 | PBRS, Euclidean + lava-proximity potential | LavaCrossing | 10 × 3M | Does the lava term reduce deaths, or also hinder crossing? | Planned (core) |
| EXP-020 | 5 more baseline seeds | both | +5 × 3M | Baseline for a 10-seed comparison | Planned (core) |
| EXP-020 | PBRS, BFS potential | both | 10 × 3M | Does the Euclidean dead end mislead the agent? | Planned (extension) |
| EXP-020 | Shifted potential 1 − d/D | LavaCrossing | 10 × 3M | Fewer early deaths, at what cost? | Planned (extension) |
| EXP-020 | Naive control | both | 10 × 3M | Why does the potential form matter? | Planned (extension) |
| EXP-021 | 5×5 view × shaping | both | to be fixed | Does shaping help more with a smaller view? | Planned |

The core conditions answer the main question (RQ1); the extensions refine the answer (RQ2, RQ3). The core
conditions run first, and if not all conditions can be run, the extensions are dropped first. The baseline
has 5 seeds per task; the main comparison is planned with 10, because the LavaCrossing baseline varies
strongly between seeds (final-success SD 0.29). The final number of seeds is fixed in the pre-registration.

Before EXP-020 starts, a pre-registration fixes the conditions, the lava penalty c, the seeds, the metrics,
the decision rules and the commit hash.

## 7. Metrics

The primary metrics were pre-registered before the EXP-011 baseline and stay the same for all later
experiments. Secondary metrics marked † were added after the baseline analysis, before any shaping run.
Each metric is computed per run.

| Metric | Meaning | Type |
|---|---|---|
| Final success rate | Stochastic policy, 100 held-out episodes (seeds 20000–20099) | Primary |
| Area under the success curve | Mean periodic-evaluation success from 0 to 3M (0–1); higher = faster learning | Primary |
| Steps to 50% / 80% success | First periodic evaluation at or above the threshold; runs that never reach it are counted separately | Primary |
| Greedy final success | Deterministic policy, same 100 episodes | Secondary |
| Death and timeout rates | Why the agent fails | Secondary |
| Successful-episode length | How direct the successful paths are | Secondary |
| Greedy loop rate | Fraction of greedy episodes that revisit a state (cell, direction) | Secondary |
| Revisit ratio | Fraction of steps that end in an already visited state (stochastic) | Secondary |
| Never-crossed timeouts † | Fraction of timed-out episodes that never cross the obstacle line | Secondary |
| First 500k steps † | Success, death and timeout rates of training episodes; tests H-lava-a and H-offset | Secondary |
| Original and shaped return | Logged separately; evaluation uses the original reward only | Control |

Periodic evaluation: every 20k steps, 20 episodes on fixed held-out seeds (10000–10019), stochastic policy.

## 8. Statistics

- Per run, then mean ± SD over seeds.
- Main comparison (shaping against baseline, per task): mean with a 95% bootstrap confidence interval and
  the probability of improvement, following Agarwal et al. (2021).
- With 5 vs 5 seeds, the exact two-sided permutation test has a smallest possible p of 0.008. Such tests are
  reported as exploratory only, without a multiple-comparison correction.

## 9. Literature (workflow step 7)

To read and verify before the paper cites them:

| Reference | Why it matters | Status |
|---|---|---|
| Ng, Harada & Russell (1999) | Policy invariance of potential-based shaping; the cycle argument against naive bonuses | To read |
| Wiewiora (2003), JAIR 19 | PBRS is equivalent to Q-value initialization; background for H-view | Existence checked; to read |
| Grześ (2017), AAMAS | Reward shaping in episodic RL: terminal potentials must be zero | Existence checked; to read |
| Müller & Kudenko (2025), arXiv:2502.01307 | Shifting a potential; background for H-offset | Existence checked; to read |
| Agarwal et al. (2021), NeurIPS | Reliable evaluation with few seeds (bootstrap CI, probability of improvement) | To verify |
| Pinto et al. (2018) | Asymmetric actor-critic: a critic with privileged state; must be compared with H-view | To read |
| Devlin & Kudenko (2012); Raileanu (2020) | Candidate references; relevance to be checked | To verify |

Every entry of the bibliography is checked against its primary source before it is used.

## 10. Threats to validity and limitations

- **Reward normalization.** VecNormalize rescales the shaped reward, so the effective shaping scale is not
  exactly the nominal one. It is kept identical in all conditions.
- **Memoryless policy.** A recurrent policy (e.g. an LSTM) is another known remedy for loops under partial
  observability. It is not studied, because it changes the agent, not the reward. It is an alternative
  explanation to discuss in the paper.
- **Small maps.** On 9×9 maps a 5×5 view already covers much of the grid. Whether the view effect grows on
  larger maps is not tested.
- **Hyperparameters.** The RL Zoo MiniGrid defaults are not tuned for the Crossing tasks. They are kept
  unchanged for every condition, so the comparison is fair, but absolute numbers may not be optimal.
- **Few seeds.** The baseline has 5 seeds per task, and the LavaCrossing spread is large. This is why 10
  seeds are planned for the main comparison.
- **Intermediate policies.** Behavior was analyzed for the final policies only; the 1M and 2M checkpoints
  are kept for later analysis if the paper needs it.

## 11. Order of the remaining work

1. Pre-register EXP-020: the conditions, the lava penalty c, the seeds, the metrics, the decision rules and
   the commit hash.
2. Run the core conditions of EXP-020, then the extensions.
3. Literature review in parallel with the runs; then the research gap.
4. Run EXP-021 (5×5 view × shaping).
5. Analysis with the same scripts as the baseline (`analyze_minigrid.py`, `loop_metrics.py`, `behavior.py`)
   and the statistics of section 8.
6. Write the paper.

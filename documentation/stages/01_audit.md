# Stage 1 — Audit of the existing work

**Workflow step:** 1 (audit). **Experiments:** EXP-000, EXP-000b. **Status:** done.

## Question

Before the project started there was some earlier code: a hand-written MuJoCo humanoid environment, a
trained PPO model and a few training logs. What of it can be verified, and can any of it serve as a baseline?

## Why this stage

Building on unverified code or numbers would make every later comparison unreliable. The audit decides what
is kept as evidence, what is kept only as a record, and what the baseline must be rebuilt from.

## What was done

### EXP-000: audit of the existing implementation

- Files found: a training script (PPO on a custom humanoid environment, with a commented-out Humanoid-v5
  variant), the custom environment (`custom_env.py`, `my_robot.xml`, a 16-actuator humanoid with two
  obstacles), viewers and smoke tests, one saved PPO model and four TensorBoard logs.
- The four runs were documented after the fact. None of them had a seed, an evaluation script or
  checkpoints.
- The custom environment and the saved model were evaluated with a deterministic policy:

| Policy | Episode length (steps) | Simulated time | Return | Forward displacement |
|---|---|---|---|---|
| Zero action (3 episodes) | 116–117 | 0.35 s | ~66 | 0.000 m |
| Random action (3 episodes) | 99–105 | ~0.31 s | 35–62 | −0.02 to +0.01 m |
| Trained PPO model (5 episodes) | 100–107 | ~0.31 s | 83–167 | +0.07 to +0.17 m |

Defects found in the custom environment:

1. **Standing is a terminal state.** Termination is torso z < 0.8 m, but the upright torso height with flat
   feet is 0.740 m.
2. **The robot spawns in the air** (feet 0.66 m above the floor), so every episode is a ~0.3 s fall.
3. **No time limit:** `truncated` is always false.
4. **Extreme torques:** actions are multiplied by 100 on top of gears of 100–300, up to 30,000 Nm
   (Humanoid-v5: about 40 Nm).
5. **A 3 ms control period** (no frame skip; Humanoid-v5 uses 15 ms).
6. **A hand-shaped reward:** 1 + 2 · (forward velocity) per step, and −50 when the robot falls. This was
   against the plan of starting from the original reward.
7. **Seeding is not reproducible:** reset noise uses the global `np.random`.
8. **The obstacles never matter:** they are 1.5 m away, the agent never moves more than ~0.17 m, and they are
   not in the observation.

### EXP-000b: reproducing the earlier Humanoid-v5 result

An earlier summary reported Humanoid-v5 evaluation returns of about 290–300, but no Humanoid-v5 model or
evaluation script was found anywhere on disk. The earlier configuration (4 environments, n_steps 2048,
batch 64, lr 3e-4, γ 0.99, λ 0.95, ~106k steps) was therefore re-run with seeds 0–2.

| Seed | Mean evaluation return ± SD (5 episodes) | Episode length | Termination | Forward displacement |
|---|---|---|---|---|
| 0 | 332.4 ± 8.7 | 60–65 | all fell | 0.55–0.62 m |
| 1 | 326.4 ± 28.0 | 56–68 | all fell | 0.52–0.60 m |
| 2 | 296.6 ± 14.5 | 52–58 | all fell | 0.60–0.63 m |

The survival bonus (5 per step) makes up about 89% of each return.

## Observations

- In the custom environment, episode length stayed at about 100 steps for the whole of training, while the
  mean reward rose.
- In Humanoid-v5, a return of about 300 means the robot falls after 55–65 steps (about 0.9 s), after lurching
  about 0.6 m forward.

## Interpretations

- In the custom environment, every episode is a fall of nearly fixed length, so the only thing the agent can
  improve is its forward velocity during the fall. PPO learned to lunge 7–17 cm forward while falling. The
  reward increase is real, but it is not locomotion. This is a concrete example of why a rising reward is not
  evidence of the intended behavior.
- The earlier Humanoid-v5 numbers fall inside the reproduced range for a ~100k-step model. This makes them
  plausible, but they cannot be verified because the model is gone.

## Decisions and reasons

- **The custom environment and its model are excluded as a baseline.** Reason: defects 1–8 make the task
  invalid (an upright robot is terminated). They are kept in `archive/humanoid/legacy/` as a record.
- **The earlier returns are recorded as "consistent with a ~100k-step model", not as a baseline.** Reason:
  the producing model and script do not exist.
- **The baseline is rebuilt from a standard environment with its original reward**, with a reproducible
  pipeline: own repository, pinned requirements, verified seeding, a config-driven training script, a
  separate evaluation script on fixed seeds, and a smoke test (train → save → reload → evaluate gives
  identical results).

## Outputs

`archive/humanoid/legacy/README.md` (what each legacy folder is), `archive/humanoid/legacy/` (the copied
earlier work and the EXP-000b reproduction script).

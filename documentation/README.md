# Documentation

This folder explains the study: the full plan, what was done at each stage, and why. The code, configs and
raw results are in the rest of the repository; the files here point to them.

| File | Contents |
|---|---|
| [research_plan.md](research_plan.md) | The full plan: motivation, workflow and its status, problem statement, research questions and hypotheses, method, experiments, metrics, statistics, literature, limitations |
| [decisions.md](decisions.md) | Every decision, its reason and the alternative not taken |
| [stages/01_audit.md](stages/01_audit.md) | Audit of the earlier code and results |
| [stages/02_humanoid_baseline.md](stages/02_humanoid_baseline.md) | Preliminary PPO baseline on MuJoCo Humanoid-v5 (archived) |
| [stages/03_switch_to_minigrid.md](stages/03_switch_to_minigrid.md) | Why the study moved to MiniGrid, and how |
| [stages/04_minigrid_baseline.md](stages/04_minigrid_baseline.md) | Pilot and 5-seed baseline with the original sparse reward |
| [stages/05_behavior_analysis.md](stages/05_behavior_analysis.md) | How and where the baseline fails: loops and the obstacle line |
| [stages/06_view_size.md](stages/06_view_size.md) | One environmental change (5×5 view) and the controlled comparison |
| [stages/07_shaping_method.md](stages/07_shaping_method.md) | The PBRS implementation, its rules and tests (no shaping run yet) |

## Status

| Workflow step | Status |
|---|---|
| 1. Audit | Done |
| 2. Baseline | Done |
| 3. Behavior analysis | Done (final policies) |
| 4. Change one environmental factor | Done (5×5 view) |
| 5. Controlled comparison | Done (5 seeds, exploratory) |
| 6. Problem statement | Draft |
| 7. Literature review | In progress |
| 8. Research gap | Not started |
| 9. Research questions and hypotheses | Draft |
| 10. Method | Implemented and tested; experiment not yet pre-registered |
| 11. Validation and ablations | Planned |
| 12. Paper | Not started |

## How each stage is written

Each stage file has the same parts: the question, why the stage was needed, what was done, the results,
**observations** (what was measured), **interpretations** (what it may mean; tentative), and **decisions**
(what was decided and why). Observations and interpretations are kept apart on purpose: an interpretation
is a hypothesis until an experiment tests it.

Numbers are means ± SD over seeds unless stated otherwise. Experiment IDs (EXP-0xx) match the folders in
`experiments/` and the configs in `configs/`.

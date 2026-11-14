# Reward shaping for sparse-reward navigation in MiniGrid

Bachelor research project by Nodir Habibullaev.

The study follows an experiment-first plan: train a standard PPO baseline with the original sparse
reward, analyse how it behaves and fails, change one environmental factor, and then test
potential-based reward shaping (PBRS) against that baseline.

The full plan, what was done at each stage and why, and every decision with its reason are in
[`documentation/`](documentation/README.md).

## Setup

- **Tasks:** `MiniGrid-SimpleCrossingS9N1-v0` (one wall line with one gap) and
  `MiniGrid-LavaCrossingS9N1-v0` (one lava line with one gap; stepping into lava ends the episode).
  9×9 grid, start at (1, 1), goal at (7, 7), at most 324 steps per episode.
- **Agent:** Stable-Baselines3 PPO with the RL Zoo MiniGrid defaults (`FlatObsWrapper`, 8 parallel
  environments, γ = 0.99), 3M environment steps per run.
- **Evaluation:** the original reward only. The primary metric is the success rate of the final policy on
  100 held-out episodes (seeds 20000–20099) with actions sampled from the policy. The greedy
  (deterministic) policy is evaluated on the same episodes as a secondary metric. During training, the
  policy is evaluated every 20k steps on 20 episodes.
- **Success, death and timeout** are read from the environment state, not inferred from the return.

## Experiments

| ID | What | Seeds × steps |
|---|---|---|
| EXP-010 | Pilot on both tasks | 1 × 1M |
| EXP-011 | Baseline: original sparse reward, 7×7 egocentric view | 5 × 3M per task |
| EXP-012 | One environmental change: 5×5 view, everything else identical | 5 × 3M per task |
| EXP-020 | PBRS: Euclidean, BFS (geodesic), Euclidean + lava and a naive control (core done); shifted potential as an extension (not started) | 10 × 3M per condition |

Final policies, mean ± SD over 5 seeds (100 held-out episodes per seed):

| Task | View | Success (sampled actions) | Success (greedy) | Lava deaths | Timeouts |
|---|---|---|---|---|---|
| SimpleCrossing | 7×7 | 1.00 ± 0.00 | 0.97 ± 0.05 | – | 0.00 |
| SimpleCrossing | 5×5 | 0.97 ± 0.07 | 0.86 ± 0.16 | – | 0.03 |
| LavaCrossing | 7×7 | 0.76 ± 0.29 | 0.62 ± 0.37 | 0.02 | 0.22 |
| LavaCrossing | 5×5 | 0.89 ± 0.07 | 0.76 ± 0.10 | 0.00 | 0.11 |

With 5 seeds, the 5×5 view shows no detectable difference in final success, area under the learning curve
or steps to 50%/80% success. On LavaCrossing, most failures are timeouts in which the agent never crosses
the lava line; every greedy failure is a loop.

**EXP-020 (core), 10 seeds per condition, 7×7 view.** AUC is the area under the success curve, the
pre-registered confirmatory metric (nine comparisons, exact permutation tests, Holm-corrected):

| Task | Sparse | Euclidean | BFS (geodesic) | Euclidean + lava | Naive control |
|---|---|---|---|---|---|
| SimpleCrossing, AUC | 0.89 ± 0.10 | 0.91 ± 0.08 | 0.88 ± 0.13 | – | 0.00 ± 0.00 |
| SimpleCrossing, final success | 1.00 ± 0.00 | 0.97 ± 0.02 | 0.98 ± 0.02 | – | 0.00 ± 0.00 |
| LavaCrossing, AUC | 0.56 ± 0.32 | 0.45 ± 0.35 | 0.84 ± 0.09 | 0.47 ± 0.39 | 0.00 ± 0.00 |
| LavaCrossing, final success | 0.76 ± 0.34 | 0.61 ± 0.40 | 0.93 ± 0.08 | 0.64 ± 0.34 | 0.00 ± 0.00 |

- On LavaCrossing, the BFS potential learns faster than the Euclidean potential (AUC +0.39 [+0.18, +0.60],
  Holm p = 0.014) and than the sparse baseline (+0.27 [+0.09, +0.48], Holm p = 0.0499, at the threshold).
- The Euclidean potential shows no detectable difference from the sparse baseline on either task.
- The naive (non-potential) bonus prevents learning in all 20 runs; its agents loop.
- On SimpleCrossing, both potentials lower final success slightly (by about 0.03).

Pre-registration: [`documentation/stages/08_shaping_experiment.md`](documentation/stages/08_shaping_experiment.md);
results and hypothesis checks: [`documentation/stages/09_shaping_results.md`](documentation/stages/09_shaping_results.md).
The next experiment, EXP-021 (the same potentials with a 5×5 view), is pre-registered in
[`documentation/stages/10_view_shaping_experiment.md`](documentation/stages/10_view_shaping_experiment.md).

Figures: `results/figures/exp011_vs_exp012_success.png` and `results/figures/exp020_success.png` (learning
curves) and
`results/figures/exp011_exp012_paths.png` (where timed-out episodes spend their steps). Short videos of
greedy failures are in `results/videos/`. Per-run tables are in `results/tables/`.

## Repository layout

```
configs/      one YAML config per experiment and task
src/          training, evaluation, analysis and the PBRS wrapper (shaping.py)
tests/        pytest tests for outcome detection and shaping
experiments/  per-run configs, periodic and final evaluations (one folder per run)
results/      tables, figures, loop metrics and videos
archive/      an earlier preliminary study on MuJoCo Humanoid-v5 (see archive/humanoid/README.md)
```

Not included: trained models and checkpoints, TensorBoard logs, per-episode training monitors, training
logs and machine-specific run metadata. The repository copies of `eval.csv` omit the wall-clock column.
Re-running training regenerates all of these.

## Reproducing

Python 3.9.7, CPU only.

```bash
pip install -r requirements.txt
python -m pytest tests

# one run, then the greedy evaluation of its final policy
# (train.py refuses to overwrite a run folder, so write new runs outside experiments/)
python src/train.py --config configs/exp011_lavacrossing.yaml --seed 0 --out runs/LavaCrossingS9N1_sparse_3M_s0
python src/evaluate.py --run-dir runs/LavaCrossingS9N1_sparse_3M_s0 \
    --episodes 100 --seed-offset 20000 --out runs/LavaCrossingS9N1_sparse_3M_s0/final_eval_greedy.json

# analysis (needs the full run folders written by train.py)
python src/analyze_minigrid.py experiments/EXP-011 experiments/EXP-012 --name exp011_vs_exp012
python src/permutation_test.py results/tables/exp011_vs_exp012_runs.csv sparse view5_sparse
python src/loop_metrics.py experiments/EXP-011 experiments/EXP-012 --name exp011_exp012
python src/behavior.py experiments/EXP-011 experiments/EXP-012 --name exp011_exp012

# shape of the Euclidean and BFS (geodesic) potentials on 1000 maps (no training needed)
python src/potential_shape.py --maps 1000

# EXP-020: the pre-registered comparisons, then the loop, line-crossing and hypothesis checks
python src/analyze_minigrid.py experiments/EXP-011 experiments/EXP-020 --name exp020
python src/compare_conditions.py results/tables/exp020_runs.csv --family exp020
python src/loop_metrics.py experiments/EXP-011 experiments/EXP-020 --name exp020
python src/behavior.py experiments/EXP-011 experiments/EXP-020 --name exp020
python src/hypothesis_checks.py --name exp020
```

The launchers `src/run_exp0*.sh` run every seed of an experiment in parallel. They write into
`experiments/EXP-0*/` and use the Python path of the original machine (`PY` in the script).

# Archived: preliminary study on MuJoCo Humanoid-v5

Moved here so that the working folders hold only the Minigrid study. Nothing was deleted.
The full record is Part 2 of the research log, which is not part of this repository.
Models, checkpoints, TensorBoard and training logs, and run metadata are kept locally, not in the repository.

- `configs/` — EXP-001 (benchmark check, config selection) and EXP-002 (5 seeds x 10M baseline) configs
- `experiments/` — EXP-001 and EXP-002 runs: logs, eval curves, final evaluations, models and checkpoints
- `src/` — Humanoid-only scripts: `gait_stats.py`, `watch.py`, `record_video.py`, `run_exp001.sh`, `run_exp002.sh`
- `results/` — EXP-001 / EXP-002 tables and figures
- `legacy/` — the pre-project prototype (custom env and early runs), not valid as results

The scripts import `common.py` from the main `src/`, so run them from the project root with
`PYTHONPATH=src`, e.g. `PYTHONPATH=src mjpython archive/humanoid/src/watch.py --run-dir archive/humanoid/experiments/EXP-002/C3_zoo_10M_s2`.
The `run_exp00*.sh` launchers expect their original location and are kept as a record only.

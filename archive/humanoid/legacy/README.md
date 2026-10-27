# Legacy material (pre-project work, kept for the record)

Nothing here is a baseline or a valid experimental result. The analysis (EXP-000, EXP-000b) is in the research log, which is not part of this repository.
Models, TensorBoard logs and run outputs are kept locally, not in the repository.

| Folder | What it is | Status |
|---|---|---|
| `custom_env_prototype/` | Hand-written 16-actuator humanoid (`my_robot.xml`), its Gymnasium wrapper, a PPO model and 3 TensorBoard runs. Copied from `~/mujoco_project`; the originals are untouched. | **Invalid environment.** Standing upright is a terminal state, and every episode is a ~0.3 s fall. |
| `humanoid_v5_smoke_*/` | TensorBoard log of the only historical Humanoid-v5 PPO run (4 envs, 106k steps, unseeded). | The model was lost. |
| `exp000b_repro_historical/` | EXP-000b: the historical config re-run with seeds 0–2 (106k steps each), with models, eval JSON and TensorBoard logs. | A verification run, not a baseline. |

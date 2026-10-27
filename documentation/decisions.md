# Decisions and their reasons

Every decision that shaped the study, grouped by the stage it belongs to. Details are in the stage files.

| # | Decision | Reason | Alternative not taken | Stage |
|---|---|---|---|---|
| 1 | Exclude the earlier custom humanoid environment and its model as a baseline | Eight defects; an upright robot is terminated, so a rising reward came from lunging during a fall, not from walking | Build on the existing model | [1](stages/01_audit.md) |
| 2 | Record the earlier Humanoid-v5 returns as "consistent with a ~100k-step model", not as a baseline | The model and evaluation script no longer exist; a re-run reproduced the range | Cite the old numbers | [1](stages/01_audit.md) |
| 3 | Check the pipeline against published PPO numbers before choosing a configuration | A bug in the baseline would make any later "improvement" meaningless | Trust the pipeline | [2](stages/02_humanoid_baseline.md) |
| 4 | Choose the Humanoid baseline configuration by a rule fixed before the results (highest 3M return) | Avoids choosing the configuration that suits a later story | Choose after looking at curves | [2](stages/02_humanoid_baseline.md) |
| 5 | Move the study from Humanoid to MiniGrid | MiniGrid fits the study (navigation, limited view, sparse reward, lightweight); Humanoid is dense-reward locomotion with full state and heavy compute, and its baseline did not walk | Continue with Humanoid, or a simpler robot | [3](stages/03_switch_to_minigrid.md) |
| 6 | MiniGrid 3.0.0, not Safety-Gymnasium | Compatible with the existing Python and gymnasium; Safety-Gymnasium pins older versions | A separate environment for Safety-Gymnasium | [3](stages/03_switch_to_minigrid.md) |
| 7 | Tasks SimpleCrossingS9N1 and LavaCrossingS9N1 | Sparse-reward navigation around an obstacle line; harmless wall vs deadly lava, so the two tasks differ in one point | FourRooms, DoorKey | [3](stages/03_switch_to_minigrid.md) |
| 8 | RL Zoo minigrid-defaults, unchanged, for every condition | A published configuration; no tuning toward one outcome | Tune hyperparameters per task | [3](stages/03_switch_to_minigrid.md) |
| 9 | Run a 1-seed pilot before designing the baseline | Measure the cost, the curve shape and the evaluation mode before fixing the design | Design the baseline blind | [4](stages/04_minigrid_baseline.md) |
| 10 | Stochastic evaluation as the primary metric, greedy as secondary | The greedy, memoryless policy loops and understates what was learned | Greedy evaluation only | [4](stages/04_minigrid_baseline.md) |
| 11 | 3M steps per run | In the pilot, SimpleCrossing was still rising at 1M and LavaCrossing reached only 0.13 | Keep 1M | [4](stages/04_minigrid_baseline.md) |
| 12 | 5 seeds per task for the baseline | Several seeds show the spread between runs; 5 keep the cost affordable. The main comparison is planned with more (decision 30) | One seed | [4](stages/04_minigrid_baseline.md) |
| 13 | Pre-register metrics and decision rules before each experiment | Prevents choosing metrics or rules after seeing results | Decide afterwards | [4](stages/04_minigrid_baseline.md) |
| 14 | Add loop metrics as a disclosed addendum, before the final analysis | The pilot showed looping; what had been seen was disclosed | Add them after the results | [4](stages/04_minigrid_baseline.md) |
| 15 | Accept the baseline unchanged; LavaCrossing "solved by 4 of 5 seeds"; compare SimpleCrossing by sample efficiency | Applying the pre-registered rules | Re-run or re-tune | [4](stages/04_minigrid_baseline.md) |
| 16 | Read outcomes from the environment state, not the reward sign | With shaping, a lava step can have a positive reward | Reward-sign rule | [5](stages/05_behavior_analysis.md) |
| 17 | Analyze where the agent fails (loops, line crossing) before fixing the shaping conditions | The expected effect of each potential should be stated against the failure that actually happens | Run the planned potentials directly | [5](stages/05_behavior_analysis.md) |
| 18 | Loop metrics and the never-crossed fraction are secondary metrics for every shaping condition | They measure the main failure that shaping should fix | Report success only | [5](stages/05_behavior_analysis.md) |
| 19 | Analyze only the final policies for now | Enough to answer the question; checkpoints are kept for later | Analyze every checkpoint | [5](stages/05_behavior_analysis.md) |
| 20 | Do not study a recurrent (memory) policy | It changes the agent, not the reward; discussed as an alternative explanation | Add an LSTM condition | [5](stages/05_behavior_analysis.md) |
| 21 | The environmental factor is the view size (5×5 instead of 7×7) | The study is about a limited egocentric view, so its size is the natural factor; also the reference for "5×5 × shaping" | Another factor | [6](stages/06_view_size.md) |
| 22 | Run the view-size step now, with 5 seeds | Postponing it would skip workflow step 4; a 1-seed pilot would not allow a comparison | Postpone it after shaping | [6](stages/06_view_size.md) |
| 23 | Treat 5-seed significance tests as exploratory | With 5 vs 5 seeds the smallest possible p is 0.008, and 16 tests were run | Report them as confirmatory | [6](stages/06_view_size.md) |
| 24 | Fixed PBRS rules: PPO's γ, Φ = 0 after termination, normal formula on truncation, fixed scale, shaping only in training, identical normalization | Keeps the PBRS guarantee and a fair comparison | Tune these per condition | [7](stages/07_shaping_method.md) |
| 25 | Add a path-length (BFS) potential as an extension condition | It follows the real path through the gap; the behavior analysis later showed that the Euclidean potential peaks at a dead end next to the line | Euclidean only | [5](stages/05_behavior_analysis.md), [7](stages/07_shaping_method.md) |
| 26 | Normalize the BFS potential by the start's BFS distance | Both distance potentials then give the same total discounted shaping signal; only its distribution over steps differs | Normalize by the largest distance on the map | [7](stages/07_shaping_method.md) |
| 27 | State the expected risk of the lava-proximity potential (H-lava-b) before running it | The gap lies in the lava line, so the penalty also hits the only way across | Run it without a stated expectation | [5](stages/05_behavior_analysis.md) |
| 28 | Add a shifted potential Φ = 1 − d/D (d = distance to the goal, D = its value at the start) as an ablation (extension) | It penalizes dying, at the cost of also penalizing the goal step; reported as a trade-off | — | [plan](research_plan.md) |
| 29 | Add a naive, non-potential control (extension) that uses the real cell's potential at termination | Shows why the potential form matters; the terminal rule must not be what is tested | No control | [7](stages/07_shaping_method.md) |
| 30 | Plan 10 seeds per condition for the main comparison | More reliable comparisons with few runs; the baseline then showed LavaCrossing final success from 0.30 to 1.00 across seeds | Keep 5 | [plan](research_plan.md) |
| 31 | The core conditions (Euclidean; Euclidean + lava; 5 more baseline seeds) run first; the extensions follow and are dropped first if not all conditions can be run | The core conditions answer the main question (RQ1); the extensions refine the answer | Run everything at once | [plan](research_plan.md) |
| 32 | Commit before every launch and record the hash; verify that earlier runs are reproduced by the current code | Every result can be traced to its exact code | Untracked code | [7](stages/07_shaping_method.md) |
| 33 | Keep models, checkpoints, raw training logs and machine-specific run records out of the repository | Size and machine-specific content; all are regenerated by re-running training | Commit everything | [README](../README.md) |

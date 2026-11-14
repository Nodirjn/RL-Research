"""RQ4 interaction contrasts of EXP-021 (src/view_interaction.py) and its 5×5 hypothesis checks.

Run from the project root: python -m pytest tests
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hypothesis_checks import CHECKS, CHECKS_BY_NAME, CHECKS_VIEW5  # noqa: E402
from view_interaction import CONFIRMATORY, PLAN, run_plan, view_rows  # noqa: E402

TASKS = ("SimpleCrossingS9N1", "LavaCrossingS9N1")
# Mean steps after crossing per condition: shaping shortens them by 4 with the 5×5 view, not with the 7×7 view.
AFTER_CROSS = {"sparse": 6.0, "euclidean": 6.0, "bfs": 6.0,
               "view5_sparse": 11.0, "view5_euclidean": 7.0, "view5_bfs": 7.0}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(tables, name, conditions):
    rng = np.random.default_rng(len(name))
    runs, loops, crossing = [], [], []
    for task in TASKS:
        for cond in conditions:
            for seed in range(10):
                run = f"{task}_{cond}_s{seed}"
                runs.append({"run": run, "task": task, "condition": cond, "seed": seed,
                             "auc": 0.5 + 0.05 * rng.normal(), "final_succ": 0.9, "steps_50": 1e6, "steps_80": ""})
                loops.append({"run": run, "revisit_ratio": 0.1, "greedy_loop_rate": 0.2})
                crossing.append({"run": run, "seed": seed, "greedy_timeouts": 4, "greedy_timeouts_never_crossed": 3,
                                 "stochastic_timeouts": 5, "stochastic_timeouts_never_crossed": 2,
                                 "greedy_succ_steps_after_cross": 6.0, "stochastic_succ_steps_to_cross": 12.0,
                                 "stochastic_succ_steps_after_cross": AFTER_CROSS[cond] + 0.5 * rng.normal()})
    write(tables / f"{name}_runs.csv", runs)
    write(tables / f"{name}_loops.csv", loops)
    write(tables / f"{name}_crossing.csv", crossing)


@pytest.fixture
def rows(tmp_path, monkeypatch):
    tables = tmp_path / "results" / "tables"
    write_tables(tables, "v7", ("sparse", "euclidean", "bfs"))
    write_tables(tables, "v5", ("view5_sparse", "view5_euclidean", "view5_bfs"))
    monkeypatch.chdir(tmp_path)
    return view_rows("v7", "v5")


def test_after_cross_timeouts_are_the_timeouts_that_crossed_as_a_share_of_all_episodes(rows):
    assert all(r["after_cross_timeout_rate"] == pytest.approx(0.03) for r in rows)


def test_plan_has_four_confirmatory_contrasts_at_the_bonferroni_level(rows):
    results = run_plan(rows)
    confirmatory = [r for r in results if r["kind"] == "confirmatory"]
    assert len(confirmatory) == 4 and all(r["metric"] == CONFIRMATORY for r in confirmatory)
    assert {(r["task"], r["B5"]) for r in confirmatory} == {(t, f"view5_{s}") for t in TASKS
                                                            for s in ("euclidean", "bfs")}
    assert all(r["level"] == pytest.approx(0.9875) for r in confirmatory)
    assert all(r["level"] == 0.95 for r in results if r["kind"] != "confirmatory")
    assert len(results) == 4 * len(PLAN)  # every metric is present in the synthetic tables
    for r in confirmatory:  # the built-in interaction of -4 steps is found
        assert r["interaction"] == pytest.approx(-4.0, abs=0.6) and r["ci_high"] < 0
        assert (r["n_a7"], r["n_b7"], r["n_a5"], r["n_b5"]) == (10, 10, 10, 10)


def test_view5_checks_are_the_exp020_checks_with_both_conditions_at_5x5():
    assert CHECKS_BY_NAME == {"exp020": CHECKS, "exp021": CHECKS_VIEW5}
    exp020 = {h: (tasks, a, b, metrics) for h, tasks, a, b, metrics in CHECKS}
    for h, tasks, a, b, metrics in CHECKS_VIEW5:
        assert (tasks, f"view5_{exp020[h][1]}", f"view5_{exp020[h][2]}", metrics) == (exp020[h][0], a, b,
                                                                                         exp020[h][3])
    assert {h for h, *_ in CHECKS_VIEW5} == {"H-dead-end", "H-lava-a"}

"""Exploratory hypothesis checks (src/hypothesis_checks.py).

Run from the project root: python -m pytest tests
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hypothesis_checks import CHECKS, merged_rows  # noqa: E402


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_never_crossed_is_a_fraction_of_the_timeouts_and_empty_without_timeouts(tmp_path, monkeypatch):
    tables = tmp_path / "results" / "tables"
    runs = [{"run": f"r{i}", "task": "LavaCrossingS9N1", "condition": "euclidean", "seed": i, "auc": 0.5}
            for i in range(2)]
    write(tables / "x_runs.csv", runs)
    write(tables / "x_loops.csv", [{**r, "auc": 0.9, "revisit_ratio": 0.1 * i} for i, r in enumerate(runs)])
    write(tables / "x_crossing.csv", [
        {"run": "r0", "seed": 0, "greedy_timeouts": 8, "greedy_timeouts_never_crossed": 6,
         "stochastic_timeouts": 4, "stochastic_timeouts_never_crossed": 1},
        {"run": "r1", "seed": 1, "greedy_timeouts": 0, "greedy_timeouts_never_crossed": 0,
         "stochastic_timeouts": 2, "stochastic_timeouts_never_crossed": 2},
    ])
    monkeypatch.chdir(tmp_path)
    r0, r1 = merged_rows("x")
    assert (r0["greedy_never_crossed"], r0["stochastic_never_crossed"]) == (0.75, 0.25)
    assert (r0["greedy_never_crossed_rate"], r0["stochastic_never_crossed_rate"]) == (0.06, 0.01)
    assert r1["greedy_never_crossed"] == ""  # no timeouts: no value, dropped by compare()
    assert r1["stochastic_never_crossed"] == 1.0
    assert r0["auc"] == "0.5"  # a column already in the runs table is kept, not overwritten
    assert r1["revisit_ratio"] == "0.1"


def test_checks_are_the_pre_registered_hypotheses_and_comparisons():
    pairs = {(h, a, b) for h, _, a, b, _ in CHECKS}
    assert pairs == {("H-dead-end", "euclidean", "bfs"), ("H-lava-a", "sparse", "euclidean"),
                     ("H-lava-b", "euclidean", "euclidean_lava"), ("H-naive", "euclidean", "naive")}
    lava_only = {h for h, tasks, *_ in CHECKS if tasks == ("LavaCrossingS9N1",)}
    assert lava_only == {"H-lava-a", "H-lava-b"}

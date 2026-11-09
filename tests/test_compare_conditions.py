"""Statistics of the pre-registered comparisons (src/compare_conditions.py).

Run from the project root: python -m pytest tests
"""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import permutation_test  # noqa: E402
from compare_conditions import (FAMILIES, bootstrap, compare, holm, permutation_p,  # noqa: E402
                                probability_of_improvement, run_family)

FIELDS = ("run", "task", "condition", "seed", "auc", "final_succ", "steps_50", "steps_80")


def test_holm_matches_a_hand_computed_example():
    # sorted 0.005, 0.01, 0.03, 0.04 -> 4 * 0.005, 3 * 0.01, 2 * 0.03, max(1 * 0.04, 0.06)
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])  # capped at 1


def test_vectorized_permutation_test_equals_the_baseline_one():
    rng = np.random.default_rng(2)
    cases = [(rng.normal(0, 1, 5), rng.normal(0.5, 1, 5)), (rng.normal(0, 1, 4), rng.normal(2, 1, 6)),
             (np.array([0.0, 1.0, 1.0, 0.5, 1.0]), np.array([1.0, 1.0, 0.5, 1.0, 1.0]))]  # ties
    for a, b in cases:
        assert permutation_p(a, b) == pytest.approx(permutation_test.permutation_p(a, b))
    assert permutation_p(np.zeros(5), np.ones(5)) == pytest.approx(2 / 252)  # the smallest p for 5 against 5


def test_probability_of_improvement_counts_ties_as_half():
    assert probability_of_improvement(np.array([0.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(0.75)
    assert probability_of_improvement(np.array([0.0, 0.0]), np.array([1.0, 2.0])) == pytest.approx(1.0)


def test_bootstrap_interval_separates_clear_differences_and_is_reproducible():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0.4, 0.05, 10), rng.normal(0.7, 0.05, 10)
    (lo, hi), (poi_lo, poi_hi) = bootstrap(a, b)
    assert lo < b.mean() - a.mean() < hi and lo > 0
    assert poi_lo == pytest.approx(1.0) and poi_hi == pytest.approx(1.0)
    assert bootstrap(a, b) == ((lo, hi), (poi_lo, poi_hi))  # fixed seed
    (lo, hi), _ = bootstrap(a, a.copy())
    assert lo < 0 < hi


def test_steps_to_threshold_never_reached_count_as_the_budget():
    rows = [{"task": "T", "condition": "A", "steps_50": ""}, {"task": "T", "condition": "A", "steps_50": "1000000"},
            {"task": "T", "condition": "B", "steps_50": "2000000"}]
    result = compare(rows, "T", "A", "B", "steps_50")
    assert result["mean_a"] == pytest.approx(2e6) and result["n_a"] == 2


def test_the_exp020_family_has_nine_comparisons_with_a_config_for_every_condition():
    family = FAMILIES["exp020"]
    assert len(family) == 9 and len(set(family)) == 9
    for task, a, b in family:
        for cond in (a, b):
            name = "simplecrossing" if task.startswith("Simple") else "lavacrossing"
            assert (ROOT / "configs" / f"exp020_{name}_{cond}.yaml").exists()


def synthetic_rows():
    rng = np.random.default_rng(0)
    rows = []
    for task, _, _ in FAMILIES["exp020"]:
        for cond, shift in (("sparse", 0.0), ("euclidean", 0.1), ("bfs", 0.3), ("naive", -0.1), ("euclidean_lava", 0.0)):
            for seed in range(10):
                if any(r["task"] == task and r["condition"] == cond and r["seed"] == seed for r in rows):
                    continue
                auc = float(np.clip(0.5 + shift + rng.normal(0, 0.05), 0, 1))
                rows.append({"run": f"{task}_{cond}_s{seed}", "task": task, "condition": cond, "seed": seed,
                             "auc": str(auc), "final_succ": str(min(1.0, auc + 0.2)),
                             "steps_50": "" if auc < 0.45 else str(int(3e6 * (1 - auc))), "steps_80": ""})
    return rows


def test_family_tests_only_auc_and_applies_holm_to_all_nine():
    results = run_family(synthetic_rows(), "exp020")
    tested = [r for r in results if r["p"] is not None]
    assert len(results) == 9 * 4 and len(tested) == 9
    assert all(r["metric"] == "auc" for r in tested)
    assert [r["p_holm"] for r in tested] == pytest.approx(list(holm([r["p"] for r in tested])))
    bfs_vs_sparse = next(r for r in tested if r["A"] == "sparse" and r["B"] == "bfs")
    assert bfs_vs_sparse["p"] < 1e-4 and bfs_vs_sparse["ci_low"] > 0.2  # a shift of 0.3 with SD 0.05


def test_command_line_writes_the_comparison_table(tmp_path):
    table = tmp_path / "exp_runs.csv"
    rows = synthetic_rows()
    with open(table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    subprocess.run([sys.executable, str(ROOT / "src" / "compare_conditions.py"), str(table), "--family", "exp020",
                    "--pairs", "euclidean:bfs", "--metrics", "auc"], check=True, capture_output=True)
    out = list(csv.DictReader(open(tmp_path / "exp_runs_comparisons.csv")))
    assert len(out) == 9 * 4 + 2  # the family, then the exploratory pair on both tasks
    assert all(r["p"] == "" for r in out[36:])  # exploratory pairs are not tested

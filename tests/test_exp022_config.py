"""EXP-022 (documentation/stages/12_shifted_potential_experiment.md): the shifted BFS potential config.

Run from the project root: python -m pytest tests
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analyze_minigrid import COLORS, LABELS  # noqa: E402
from common import load_config  # noqa: E402
from shaping import PBRS  # noqa: E402

EXP021 = ROOT / "configs" / "exp021_simplecrossing_bfs.yaml"
EXP022 = ROOT / "configs" / "exp022_simplecrossing_bfs_shifted.yaml"


def test_exp022_differs_from_the_exp021_bfs_config_only_in_the_offset():
    a, b = load_config(EXP021), load_config(EXP022)
    assert a["shaping"] == {"potential": "bfs"}
    assert b["shaping"] == {"potential": "bfs", "offset": 1.0}
    assert (b["experiment"], b["condition"]) == ("EXP-022", "SimpleCrossingS9N1_view5_bfs_shifted")
    for cfg in (a, b):
        for key in ("experiment", "condition", "shaping"):
            del cfg[key]
    assert a == b


def test_the_shifted_potential_is_the_bfs_potential_plus_one_on_every_cell():
    from common import make_env
    cfg = load_config(EXP022)
    plain = PBRS(make_env(cfg), potential="bfs")
    shifted = PBRS(make_env(cfg), gamma=cfg["ppo"]["gamma"], **cfg["shaping"])
    for env in (plain, shifted):
        env.reset(seed=4)
    base = shifted.env.unwrapped
    for x in range(base.grid.width):
        for y in range(base.grid.height):
            if plain.bfs[x, y] != float("inf"):
                for env in (plain, shifted):
                    env.env.unwrapped.agent_pos = (x, y)
                assert shifted.phi() == plain.phi() + 1.0


def test_the_exp022_condition_has_a_color_and_a_label():
    name = load_config(EXP022)["condition"].replace("SimpleCrossingS9N1_", "")
    assert name in COLORS and name in LABELS

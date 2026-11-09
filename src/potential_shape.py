"""Shape of the Euclidean and the BFS (geodesic) potential on the S9N1 maps. No learning is involved.

Usage:
    python src/potential_shape.py --maps 1000

For every map (reset seeds 0 .. maps-1) and both potentials, steepest ascent is followed over the cells the
agent can stand on (no walls, no lava): from a cell, move to the 4-neighbour with the highest Phi if it is
strictly higher, else stop (ties go to the first neighbour in shaping.NEIGHBOURS). A potential whose ascent
stops before the goal has a local maximum there: a dead end that the shaping signal points into (H-dead-end).
This describes the signal only; a learning agent explores and is not greedy.

Per task and potential, over all maps:
    start_reaches_goal       fraction of maps where the ascent from the start reaches the goal
    start_side_reach_goal    fraction of start-side cells (before the obstacle line) whose ascent reaches it
    dead_end_next_to_line    among stuck ascents from the start: fraction that stop next to the line
The two tasks generate the same layout (line position and gap) for the same reset seed; only the obstacle
type differs. Both potentials treat walls and lava alike, so their shape, and this table, is the same for
both tasks: what differs is that entering lava ends the episode.
Writes results/tables/potential_shape.csv and results/figures/potentials_euclidean_vs_bfs.png (one example
map per task where the Euclidean ascent from the start gets stuck: a vertical line for SimpleCrossing, a
horizontal one for LavaCrossing, so the figure shows two different layouts).
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

from behavior import CELL_COLORS, SURFACE, TEXT, TEXT_2, obstacle_line
from common import make_env
from shaping import NEIGHBOURS, bfs_distances

ROOT = Path(__file__).resolve().parents[1]
TASKS = {  # task: (env id, line axis of the example map)
    "SimpleCrossing": ("MiniGrid-SimpleCrossingS9N1-v0", "x"),
    "LavaCrossing": ("MiniGrid-LavaCrossingS9N1-v0", "y"),
}
START = (1, 1)


def potentials(grid, goal):
    """Phi of every cell the agent can stand on, for both potentials, scaled as in shaping.PBRS."""
    free = [(x, y) for x in range(grid.width) for y in range(grid.height)
            if grid.get(x, y) is None or grid.get(x, y).type == "goal"]
    bfs = bfs_distances(grid, goal)
    d_e, d_b = np.hypot(START[0] - goal[0], START[1] - goal[1]), bfs[START]
    return {
        "euclidean": {c: -np.hypot(c[0] - goal[0], c[1] - goal[1]) / d_e for c in free},
        "bfs": {c: -bfs[c] / d_b for c in free},
    }


def ascent(phi, cell):
    path = [cell]
    while True:
        x, y = path[-1]
        options = [(x + dx, y + dy) for dx, dy in NEIGHBOURS if (x + dx, y + dy) in phi]
        best = max(options, key=phi.get)
        if phi[best] <= phi[path[-1]]:
            return path
        path.append(best)


def start_side(cell, line):
    axis, pos, _ = line
    return (cell[0] if axis == "x" else cell[1]) < pos


def analyze(env_id, example_axis, n_maps):
    env = make_env({"env_id": env_id, "wrappers": ["minigrid.wrappers.FlatObsWrapper"]})  # importing registers
    rows = {name: {"start": [], "side": [], "next_to_line": []} for name in ("euclidean", "bfs")}
    example = None
    for seed in range(n_maps):
        env.reset(seed=seed)
        base = env.unwrapped
        goal = tuple(int(v) for v in base.goal_position)
        line = obstacle_line(env)
        phis = potentials(base.grid, goal)
        for name, phi in phis.items():
            path = ascent(phi, START)
            rows[name]["start"].append(path[-1] == goal)
            side = [c for c in phi if start_side(c, line)]
            rows[name]["side"].append(np.mean([ascent(phi, c)[-1] == goal for c in side]))
            if path[-1] != goal:
                axis, pos, _ = line
                rows[name]["next_to_line"].append((path[-1][0] if axis == "x" else path[-1][1]) == pos - 1)
        if example is None and line[0] == example_axis and ascent(phis["euclidean"], START)[-1] != goal:
            example = {"seed": seed, "grid": base.grid, "goal": goal, "line": line, "phis": phis}
    return rows, example


def draw(ax, example, name, vmin):
    grid, phi = example["grid"], example["phis"][name]
    image = np.full((grid.height, grid.width), np.nan)
    for (x, y), value in phi.items():
        image[y, x] = value
    ax.imshow(image, cmap="Blues", vmin=vmin, vmax=0.0, origin="upper")
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell is not None and cell.type in ("wall", "lava"):
                ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=CELL_COLORS[cell.type],
                                       hatch="////" if cell.type == "lava" else None, edgecolor=SURFACE, lw=0))
            elif (x, y) in phi:
                dark = (phi[(x, y)] - vmin) / -vmin > 0.6
                label = "G" if (x, y) == example["goal"] else f"{phi[(x, y)]:.2f}"
                ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
                        color=SURFACE if dark else TEXT, fontweight="bold" if label == "G" else None)
    for k in range(grid.width + 1):
        ax.axhline(k - 0.5, color=SURFACE, lw=2)
        ax.axvline(k - 0.5, color=SURFACE, lw=2)
    # The ascent path is drawn as outlined cells, so the numbers stay readable; a dead end gets a thick outline.
    path = ascent(phi, START)
    for x, y in path:
        ax.add_patch(Rectangle((x - 0.38, y - 0.38), 0.76, 0.76, facecolor="none", edgecolor=TEXT, lw=1.6, zorder=3))
    if path[-1] != example["goal"]:
        x, y = path[-1]
        ax.add_patch(Rectangle((x - 0.44, y - 0.44), 0.88, 0.88, facecolor="none", edgecolor=TEXT, lw=4, zorder=4))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", type=int, default=1000)
    args = parser.parse_args()

    table, examples = [], {}
    for task, (env_id, axis) in TASKS.items():
        rows, examples[task] = analyze(env_id, axis, args.maps)
        for name, r in rows.items():
            table.append({
                "task": task, "potential": name, "maps": args.maps,
                "start_reaches_goal": round(float(np.mean(r["start"])), 4),
                "start_side_reach_goal": round(float(np.mean(r["side"])), 4),
                "stuck_from_start": len(r["next_to_line"]),
                "dead_end_next_to_line": round(float(np.mean(r["next_to_line"])), 4) if r["next_to_line"] else "",
            })
    out = ROOT / "results" / "tables" / "potential_shape.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for row in table:
        print(row)

    vmin = min(min(e["phis"][n].values()) for e in examples.values() for n in ("euclidean", "bfs"))
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 10.4), facecolor=SURFACE)
    titles = {"euclidean": "Euclidean  Φ = −d / D", "bfs": "BFS (geodesic)  Φ = −b / B"}
    for row, (task, example) in enumerate(examples.items()):
        for col, name in enumerate(("euclidean", "bfs")):
            draw(axes[row, col], example, name, vmin)
            axes[row, col].set_title(f"{task} (map seed {example['seed']})\n{titles[name]}",
                                     fontsize=10, color=TEXT, loc="left")
    handles = [
        Patch(facecolor=CELL_COLORS["wall"], label="wall"),
        Patch(facecolor=CELL_COLORS["lava"], hatch="////", edgecolor=SURFACE, label="lava"),
        Patch(facecolor="none", edgecolor=TEXT, lw=1.6, label="steepest-ascent path from the start"),
        Patch(facecolor="none", edgecolor=TEXT, lw=3, label="local maximum (dead end)"),
    ]
    fig.subplots_adjust(left=0.03, right=0.86, top=0.93, bottom=0.08, hspace=0.2, wspace=0.06)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, 0.025), ncol=4, frameon=False,
               fontsize=8.5, labelcolor=TEXT, handlelength=1.4)
    fig.text(0.45, 0.008, "The start is the top-left free cell (Φ = −1), G = goal (Φ = 0). Numbers are Φ of each cell. No learning: the shape of the signal only.",
             ha="center", fontsize=8, color=TEXT_2)
    mappable = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin, 0.0))
    bar = fig.colorbar(mappable, cax=fig.add_axes([0.89, 0.3, 0.018, 0.4]))
    bar.set_label("Potential Φ (higher = closer to the goal)", color=TEXT_2)
    bar.ax.tick_params(colors=TEXT_2, labelsize=8)
    bar.outline.set_visible(False)
    path = ROOT / "results" / "figures" / "potentials_euclidean_vs_bfs.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print("wrote", out.relative_to(ROOT), "and", path.relative_to(ROOT))


if __name__ == "__main__":
    main()

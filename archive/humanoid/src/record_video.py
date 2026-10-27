"""Record one deterministic episode of a saved policy as a GIF plus a filmstrip PNG.

Usage:
    python src/record_video.py --run-dir experiments/EXP-001/C2_cleanrl_3M_s0 [--model final|1000000]
        [--seed 20000] [--out-dir <run-dir>/videos]

The GIF plays in real time. The filmstrip shows 8 evenly spaced frames with their sim time,
so behavior can be inspected (and compared across checkpoints) without a video player.
The episode is the same one `run_episodes` would produce for this seed.
"""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

from common import load_policy

FILMSTRIP_FRAMES = 8


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default="final")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--out-dir")
    parser.add_argument("--stride", type=int, default=2, help="keep every n-th frame in the GIF")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=300)
    args = parser.parse_args()

    model, normalize_obs, cfg = load_policy(args.run_dir, args.model)
    env = gym.make(cfg["env_id"], render_mode="rgb_array", width=args.width, height=args.height)
    dt = env.unwrapped.dt

    obs, _ = env.reset(seed=args.seed)
    x0 = float(env.unwrapped.data.qpos[0])
    frames, ret, length = [env.render()], 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        policy_obs = normalize_obs(obs) if normalize_obs is not None else obs
        action, _ = model.predict(policy_obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        ret += float(reward)
        length += 1
        frames.append(env.render())
    x_displacement = float(env.unwrapped.data.qpos[0]) - x0
    env.close()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.run_dir) / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.model}_seed{args.seed}"

    gif = [Image.fromarray(f) for f in frames[::args.stride]]
    gif[0].save(out_dir / f"{name}.gif", save_all=True, append_images=gif[1:],
                duration=int(1000 * dt * args.stride), loop=0)

    picks = np.linspace(0, len(frames) - 1, FILMSTRIP_FRAMES).astype(int)
    strip = Image.new("RGB", (args.width * 4, args.height * 2))
    for i, idx in enumerate(picks):
        tile = Image.fromarray(frames[idx])
        ImageDraw.Draw(tile).text((8, 8), f"t={idx * dt:.2f}s (step {idx})", fill=(255, 255, 255))
        strip.paste(tile, ((i % 4) * args.width, (i // 4) * args.height))
    strip.save(out_dir / f"{name}_filmstrip.png")

    ending = "fell (terminated)" if terminated else "time limit (truncated)"
    print(f"{out_dir / name}.gif: return={ret:.1f} length={length} ({length * dt:.2f}s) "
          f"x_displacement={x_displacement:+.2f}m ending={ending}")


if __name__ == "__main__":
    main()

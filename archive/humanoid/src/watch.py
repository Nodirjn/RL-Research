"""Watch a saved policy live in the interactive MuJoCo viewer (real time, camera follows the robot).

Usage (macOS needs `mjpython` instead of `python` for the viewer window):
    mjpython src/watch.py --run-dir experiments/EXP-002/C3_zoo_10M_s2 [--model final|1000000]
        [--seed 20000] [--episodes 0] [--slowdown 1.0]

Each episode uses the next seed (seed, seed+1, ...), so episode k is the same one the held-out
evaluation saw. After every episode the terminal prints its return, length and ending.
Close the window (or Ctrl+C) to stop. In the viewer: drag to rotate, right-drag to pan, scroll to zoom,
Space pauses, and double-click a body then Ctrl+drag to push it.
"""
import argparse
import time

import gymnasium as gym
import mujoco
import mujoco.viewer

from common import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default="final")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--episodes", type=int, default=0, help="stop after this many episodes (0 = until closed)")
    parser.add_argument("--slowdown", type=float, default=1.0, help="2.0 = half speed")
    args = parser.parse_args()

    model, normalize_obs, cfg = load_policy(args.run_dir, args.model)
    env = gym.make(cfg["env_id"])
    m, d = env.unwrapped.model, env.unwrapped.data
    dt = env.unwrapped.dt
    print(f"{args.run_dir} ({args.model}) on {cfg['env_id']}; close the window to stop.")

    episode = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
        viewer.cam.distance = 4.0
        while viewer.is_running() and (args.episodes == 0 or episode < args.episodes):
            seed = args.seed + episode
            obs, _ = env.reset(seed=seed)  # resets the same MjData in place, so the viewer stays attached
            x0 = float(d.qpos[0])
            ret, length = 0.0, 0
            terminated = truncated = False
            viewer.sync()
            while viewer.is_running() and not (terminated or truncated):
                step_start = time.time()
                action, _ = model.predict(normalize_obs(obs) if normalize_obs is not None else obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                ret += float(reward)
                length += 1
                viewer.sync()
                time.sleep(max(0.0, dt * args.slowdown - (time.time() - step_start)))
            ending = "fell" if terminated else "time limit" if truncated else "window closed"
            print(f"episode {episode} (seed {seed}): return={ret:.1f} length={length} ({length * dt:.2f}s) "
                  f"x_displacement={float(d.qpos[0]) - x0:+.2f}m ending={ending}")
            episode += 1
            time.sleep(1.0)  # a short pause so the fall is visible before the reset
    env.close()


if __name__ == "__main__":
    main()

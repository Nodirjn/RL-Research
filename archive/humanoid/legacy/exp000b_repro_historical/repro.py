"""Reproduction check of the historical Humanoid-v5 PPO run (handover section 2B/2D).

Same hyperparameters as the historical script, but with an explicit seed so the
result is reproducible. This is NOT the official baseline.
"""
import json
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

seed = int(sys.argv[1])
out_dir = sys.argv[2]
torch.set_num_threads(1)  # several seeds run in parallel on 4 cores

env = make_vec_env("Humanoid-v5", n_envs=4, seed=seed)
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    seed=seed,
    device="cpu",
    verbose=0,
    tensorboard_log=f"{out_dir}/tb",
)
t0 = time.time()
model.learn(total_timesteps=100_000, tb_log_name=f"seed{seed}")
train_s = time.time() - t0
model.save(f"{out_dir}/model_seed{seed}")

eval_env = gym.make("Humanoid-v5")
episodes = []
for ep in range(5):
    obs, info = eval_env.reset(seed=10_000 + ep)
    x0 = eval_env.unwrapped.data.qpos[0]
    ret, n = 0.0, 0
    comps = {"reward_survive": 0.0, "reward_forward": 0.0, "reward_ctrl": 0.0, "reward_contact": 0.0}
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = eval_env.step(action)
        ret += r
        n += 1
        for k in comps:
            comps[k] += info[k]
        if terminated or truncated:
            break
    episodes.append({
        "return": round(ret, 2),
        "length": n,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "x_displacement_m": round(float(eval_env.unwrapped.data.qpos[0] - x0), 3),
        **{k: round(v, 2) for k, v in comps.items()},
    })

result = {
    "seed": seed,
    "timesteps": int(model.num_timesteps),
    "train_seconds": round(train_s, 1),
    "fps": round(model.num_timesteps / train_s),
    "episodes": episodes,
}
json.dump(result, open(f"{out_dir}/result_seed{seed}.json", "w"), indent=2)
print(json.dumps(result))

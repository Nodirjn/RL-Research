import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco


class CustomHumanoidEnv(gym.Env):
    def __init__(self):
        super(CustomHumanoidEnv, self).__init__()

        # XML modelni yuklaymiz
        self.model = mujoco.MjModel.from_xml_path("my_robot.xml")
        self.data = mujoco.MjData(self.model)

        # Action space (motorlar kuchi)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

        # Observation space
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Boshlang'ich holat
        self.data.qpos[:] += np.random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action * 100.0
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        x_velocity = self.data.qvel[0]  # Oldinga tezlik
        height = self.data.qpos[2]  # Gavda balandligi

        # Mukofotlar (Tik tursa +1, oldinga yursa tezligiga qarab ball)
        alive_reward = 1.0
        forward_reward = x_velocity * 2.0

        reward = alive_reward + forward_reward

        # Yiqilish sharti (Balandlik 0.8m dan past bo'lsa)
        terminated = height < 0.8 or height > 2.0
        if terminated:
            reward -= 50.0  # Yiqilgani uchun jazo

        truncated = False
        return obs, reward, terminated, truncated, {}
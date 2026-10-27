import time
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from custom_env import CustomHumanoidEnv


def main():
    print("--- 5-BOSQICH: Saqlangan AIning harakatini test qilish ---")

    env = CustomHumanoidEnv()

    # Modelni yuklaymiz
    model = PPO.load("humanoid_ppo_model")

    obs, _ = env.reset()

    # Passive viewer orqali simulyatsiya oynasini ochamiz
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            # AI modelidan keyingi harakatni olamiz
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                obs, _ = env.reset()

            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
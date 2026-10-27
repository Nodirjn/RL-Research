# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
#
# def main():
#     print("--- 1-BOSQICH: Humanoid robotni o'rgatish boshlanmoqda ---")
#
#     # 1. Tezroq o'rganishi uchun 4 ta parallel muhit ochamiz (Vectorized Env)
#     env_id = "Humanoid-v5"
#     num_envs = 4
#     env = make_vec_env(env_id, n_envs=num_envs)
#
#     # 2. PPO algoritmini sozlaymiz
#     model = PPO(
#         policy="MlpPolicy",
#         env=env,
#         learning_rate=3e-4,
#         n_steps=2048,
#         batch_size=64,
#         gamma=0.99,
#         gae_lambda=0.95,
#         verbose=1,
#         tensorboard_log="./humanoid_ppo_logs/"
#     )
#
#     # 3. Modelni o'rgatish (Kichik test uchun 100,000 qadam, to'liq yura olishi uchun kamida 1M-2M qadam kerak)
#     total_timesteps = 100000
#     print(f"Robot {total_timesteps} qadam davomida o'rgatilmoqda...")
#     model.learn(total_timesteps=total_timesteps)
#
#     # 4. Saqlaymiz
#     model.save("humanoid_ppo_model")
#     print("Model 'humanoid_ppo_model.zip' fayliga muvaffaqiyatli saqlandi!")
#
#     env.close()
#
# if __name__ == "__main__":
#     main()




from stable_baselines3 import PPO
from custom_env import CustomHumanoidEnv

def main():
    print("--- Shaxsiy to'siqli Humanoidni o'rgatish boshlandi ---")
    env = CustomHumanoidEnv()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        tensorboard_log="./humanoid_ppo_logs/"
    )

    total_timesteps = 1000000  # 1 Million qadam
    model.learn(total_timesteps=total_timesteps)
    model.save("humanoid_ppo_model")
    print("Model muvaffaqiyatli saqlandi!")

if __name__ == "__main__":
    main()
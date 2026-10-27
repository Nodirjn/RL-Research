import gymnasium as gym

# 1. Tayyor MuJoCo Humanoid muhitini yuklaymiz
env = gym.make("Humanoid-v5", render_mode="human")

# 2. Muhitni boshlang'ich holatga keltiramiz
obs, info = env.reset()

print("MuJoCo va Gymnasium muvaffaqiyatli ishlamoqda!")
print("Kuzatuv o'lchami (Observation shape):", obs.shape)

# 3. Robotni 200 qadam davomida tasodifiy harakatlantiramiz
for _ in range(200):
    action = env.action_space.sample()  # Tasodifiy harakat
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()

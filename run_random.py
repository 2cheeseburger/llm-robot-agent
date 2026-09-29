import gymnasium as gym
import minigrid  # registers the environments

env = gym.make("MiniGrid-Fetch-5x5-N2-v0")
obs, info = env.reset(seed=0)
print("Mission:", obs["mission"])

for step in range(30):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(step, "action:", action, "reward:", reward)
    if terminated or truncated:
        break

env.close()

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import os

algorithm = PPO
algorithm_name = f"{algorithm.__name__}_CPU"
models_dir = f"models/{algorithm_name}"
model_path = f"{models_dir}/290000.zip"

# env = gym.make('LunarLander-v2')
env = gym.make('LunarLander-v2', render_mode="human")  # continuous: LunarLanderContinuous-v2
env.metadata["render_fps"] = 120
env.reset()

# LOAD and RUN
model = algorithm.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    # print(ep)
    observation, info = env.reset()
    terminated = False
    while not terminated:
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if int(reward) > 80 or int(reward) < -80:
            print(reward)
env.close()

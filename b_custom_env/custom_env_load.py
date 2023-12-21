import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import os
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from pcustom import CustomEnv

algorithm = PPO
model_name = "PPO-custom5x5-1700939144"
models_dir = f"models/{model_name}"
model_path = f"{models_dir}/290000.zip"

env = CustomEnv(render_mode="human")
env.reset()

# LOAD and RUN
model = algorithm.load(model_path, env=env)
episodes = 2

for ep in range(episodes):
    observation, info = env.reset()
    env.render()
    terminated = False
    while not terminated:
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
env.close()

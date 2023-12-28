import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import os
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.wrappers import ImgObsWrapper
from cs2d.d_custom_minigrid.custom_minigrid_env import CustomMinigridEnv

algorithm = PPO
algorithm_name = "PPO-1703177824"
env_name = "CustomMinigrid-v1"

models_dir = f"../models/{env_name}/{algorithm_name}"
model_path = f"{models_dir}/20000.zip"
logdir = f"../logs/{env_name}"

env = CustomMinigridEnv(render_mode="human", max_steps=50)
env = ImgObsWrapper(env)
env.reset()

# LOAD and RUN
model = algorithm.load(model_path, env=env)
episodes = 10

for ep in range(episodes):
    print(f"**************{ep}")
    observation, info = env.reset()
    terminated = False
    while not terminated:
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if reward != 0:
            print(reward)
env.close()

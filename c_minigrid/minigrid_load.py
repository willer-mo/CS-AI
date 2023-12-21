import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import os
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.wrappers import ImgObsWrapper

algorithm = PPO
algorithm_name = "PPO-1700138155"
models_dir = f"models/{algorithm_name}"
model_path = f"{models_dir}/490000.zip"


env = gym.make('MiniGrid-Empty-16x16-v0', render_mode="human", max_episode_steps=6)
env = ImgObsWrapper(env)
env.metadata["render_fps"] = 30
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

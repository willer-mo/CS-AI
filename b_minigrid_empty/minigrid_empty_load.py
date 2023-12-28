import gymnasium as gym
from stable_baselines3 import PPO
import os
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.wrappers import ImgObsWrapper

################## USER PARAMETERS ##################
env_name = "MiniGrid-Empty"  # MiniGrid-Empty-5x5-v0; MiniGrid-Empty-16x16-v0
algorithm = PPO
suffix = "2"  # Sufijo del modelo guardado en la carpeta models
zip_model = "490000.zip"  # Carpeta zip del modelo
episodes = 15
max_steps = 6
#####################################################

model_name = f"{algorithm.__name__}{'_' + suffix if suffix else ''}"
models_dir = f"../models/{env_name}/{model_name}"
model_path = f"{models_dir}/{zip_model}"

env = gym.make(env_name + "-16x16-v0", render_mode="human", max_episode_steps=max_steps)
env = ImgObsWrapper(env)
env.metadata["render_fps"] = 30
env.reset()

# LOAD and RUN
model = algorithm.load(model_path, env=env)

for ep in range(episodes):
    print(f"*** Episode: {ep}")
    observation, info = env.reset()
    terminated = False
    while not terminated:
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if reward != 0:
            print(reward)
        if terminated:
            print("Terminated!!")
env.close()

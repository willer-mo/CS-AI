import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import os
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.wrappers import ImgObsWrapper
from make_shooting_minigrid_env import make_shooting_minigrid_env


################## USER PARAMETERS ##################
env_name = "ShootingMiniGrid-v3"
grid_size = 25
algorithm = PPO
suffix = "5"  # Sufijo del modelo guardado en la carpeta models
zip_model = "200000.zip"  # Carpeta zip del modelo
episodes = 10
max_steps = 10
#####################################################

model_name = f"{algorithm.__name__}{'_' + suffix if suffix else ''}"
models_dir = f"../models/{env_name}/{model_name}"
model_path = f"{models_dir}/{zip_model}"

env = make_shooting_minigrid_env(env_version=env_name, render_mode="human", max_steps=max_steps, size=grid_size, agent_start_pos=(1, 1))
#env = ImgObsWrapper(env)
env.reset()

# Sleep so that an external screen recorder has time to detect the new pygame window
time.sleep(1)

# LOAD and RUN
model = algorithm.load(model_path, env=env)

for ep in range(episodes):
    print(f"*** Episode: {ep}")
    observation, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        print("Action: ", action, "Reward: ", reward)
        if terminated:
            print("Terminated!!")
env.close()

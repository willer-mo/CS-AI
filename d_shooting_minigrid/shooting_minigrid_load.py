import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import os
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.wrappers import ImgObsWrapper
from cs2d.d_shooting_minigrid.shooting_minigrid_env import ShootingMiniGridEnv


################## USER PARAMETERS ##################
env_name = "ShootingMiniGrid-v1"
algorithm = PPO
suffix = ""  # Sufijo del modelo guardado en la carpeta models
zip_model = "20000.zip"  # Carpeta zip del modelo
episodes = 10
max_steps = 50
#####################################################

model_name = f"{algorithm.__name__}{'_' + suffix if suffix else ''}"
models_dir = f"../models/{env_name}/{model_name}"
model_path = f"{models_dir}/{zip_model}"

env = ShootingMiniGridEnv(render_mode="human", max_steps=10)
env = ImgObsWrapper(env)
env.reset()

# LOAD and RUN
model = algorithm.load(model_path, env=env)

for ep in range(episodes):
    print(f"*** Episode: {ep}")
    observation, info = env.reset()
    terminated = False
    step = 0
    while not terminated and step < 30:
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        step += 1
        if reward != 0:
            print(reward)
        if terminated:
            print("Terminated!!")
env.close()

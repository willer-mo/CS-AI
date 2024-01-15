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
suffix = "20"  # Sufijo del modelo guardado en la carpeta models
zip_model = "6020000.zip"  # Carpeta zip del modelo
episodes = 20
max_steps = 50
#####################################################

model_name = f"{algorithm.__name__}{'_' + suffix if suffix else ''}"
models_dir = f"../models/{env_name}/{model_name}"
model_path = f"{models_dir}/{zip_model}"

env = make_shooting_minigrid_env(env_version=env_name, render_mode="human", max_steps=max_steps, size=grid_size, static_walls=True, agent_start_pos=(1, 12))
#env = ImgObsWrapper(env)
env.reset()

# Sleep so that an external screen recorder has time to detect the new pygame window
time.sleep(1)

# LOAD and RUN
model = algorithm.load(model_path, env=env)


def action_to_str(act):
    if 0 <= act <= 90:
        return "Rotate {}º".format(act - 45)
    elif act in [91, 92, 93, 94]:
        if act == 91:
            return "Move forward"
        if act == 92:
            return "Move left"
        if act == 93:
            return "Move backwards"
        if act == 94:
            return "Move right"
    elif act == 95:
        return "Shoot"

rewards = []
for ep in range(episodes):
    print(f"*** Episode: {ep}")
    observation, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        # print("Action: ", action_to_str(action), "\tReward: ", reward)
        # if terminated:
            # print("Terminated!!")
        if terminated or truncated:
            rewards.append(reward)
env.close()
print(rewards)
print(max(rewards))
print(sum(rewards) / len(rewards))

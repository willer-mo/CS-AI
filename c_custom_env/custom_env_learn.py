import os
import time
from stable_baselines3 import PPO
from custom_env import CustomEnv
from cs2d.utils import set_scaffolding, write_info_file


################## USER PARAMETERS ##################
env_name = "CustomEnv_5x5"
algorithm = PPO
policy = "MlpPolicy"
device = "cpu"  # Device: cpu or cuda
max_steps = 10
timesteps_per_save = 10000
number_of_saves = 30
description = ""  # Description for the readme file
#####################################################

algorithm_name = algorithm.__name__
logdir, models_dir, model_name = set_scaffolding(
    env_name=env_name, algorithm_name=algorithm_name, policy=policy, device=device
)
t = int(time.time())

env = CustomEnv(max_episode_steps=max_steps)
env.reset()

# Training
model = algorithm('MlpPolicy', env, verbose=1, device="cpu", tensorboard_log=logdir)
# model = algorithm('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs, device="cpu", tensorboard_log=logdir)

for i in range(1, number_of_saves + 1):
    model.learn(total_timesteps=timesteps_per_save, reset_num_timesteps=False, tb_log_name=f"{model_name}")
    model.save(f"{models_dir}/{timesteps_per_save * i}")
env.close()

write_info_file(
    env_name=env_name, algorithm_name=algorithm_name, model_name=model_name, policy=policy, device=device,
    episodes=number_of_saves * timesteps_per_save, models_dir=models_dir, description=description,
    time_elapsed=int(time.time()) - t
)

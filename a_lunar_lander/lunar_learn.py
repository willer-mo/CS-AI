import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from cs2d.utils import set_scaffolding, write_info_file


################## USER PARAMETERS ##################
env_name = "LunarLander-v2"
algorithm = PPO
policy = "MlpPolicy"
device = "cpu"  # Device: cpu or cuda
timesteps_per_save = 10000
number_of_saves = 30
description = ""  # Description for the readme file
#####################################################

algorithm_name = algorithm.__name__
logdir, models_dir, model_name = set_scaffolding(
    env_name=env_name, algorithm_name=algorithm_name, policy=policy, device=device
)
t = int(time.time())

env = gym.make(env_name)
env.reset()

# Training
model = algorithm(policy, env, verbose=1, device=device, tensorboard_log=logdir)

for i in range(1, number_of_saves + 1):
    model.learn(total_timesteps=timesteps_per_save, reset_num_timesteps=False, tb_log_name=f"{model_name}")
    model.save(f"{models_dir}/{timesteps_per_save * i}")
env.close()

write_info_file(
    env_name=env_name, algorithm_name=algorithm_name, model_name=model_name, policy=policy, device=device,
    episodes=number_of_saves * timesteps_per_save, models_dir=models_dir, description=description,
    time_elapsed=int(time.time()) - t
)

import time
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from make_shooting_minigrid_env import make_shooting_minigrid_env
from cs2d.utils import set_scaffolding, write_info_file


################## USER PARAMETERS ##################
env_name = "ShootingMiniGrid-v3"
grid_size = 25
random_walls = False  # False = No walls
static_walls = True  # False = No walls
algorithm = PPO
policy = "MlpPolicy"
device = "cuda"  # Device: cpu or cuda
max_steps = 50
timesteps_per_save = 10000
number_of_saves = 700
description = "grid_size = 25, max_steps = 50, agent_position = (1, 12), target_position = random, random_walls = False, static_walls = True (all columns)"  # Description for the readme file
#####################################################


algorithm_name = algorithm.__name__
logdir, models_dir, model_name = set_scaffolding(
    env_name=env_name, algorithm_name=algorithm_name, policy=policy, device=device
)
suffix = "17"
zip_model = "3000000.zip"
model_name2 = f"{algorithm.__name__}{'_' + suffix if suffix else ''}"
models_dir2 = f"../models/{env_name}/{model_name2}"
model_path2 = f"{models_dir2}/{zip_model}"

t = int(time.time())


env = make_shooting_minigrid_env(env_version=env_name, max_steps=max_steps, size=grid_size, random_walls=random_walls, static_walls=static_walls, agent_start_pos=(1, 12))
env.reset()

# Training
# model = algorithm(policy, env, verbose=1, device=device, tensorboard_log=logdir)
model = algorithm.load(model_path2, policy=policy, env=env, verbose=1, device=device, tensorboard_log=logdir)

for i in range(1, number_of_saves + 1):
    model.learn(total_timesteps=timesteps_per_save, reset_num_timesteps=False, tb_log_name=f"{model_name}")
    model.save(f"{models_dir}/{timesteps_per_save * i}")
env.close()

write_info_file(
    env_name=env_name, algorithm_name=algorithm_name, model_name=model_name, policy=policy, device=device,
    episodes=number_of_saves * timesteps_per_save, models_dir=models_dir, description=description,
    time_elapsed=int(time.time()) - t
)

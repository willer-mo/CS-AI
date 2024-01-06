import time
from minigrid.wrappers import ImgObsWrapper
import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from make_shooting_minigrid_env import make_shooting_minigrid_env
from cs2d.utils import set_scaffolding, write_info_file


################## USER PARAMETERS ##################
env_name = "ShootingMiniGrid-v1"
grid_size = 25
random_walls = True
algorithm = PPO
policy = "CnnPolicy"
device = "cuda"  # Device: cpu or cuda
max_steps = 100
timesteps_per_save = 10000
number_of_saves = 300
description = "grid_size = 25, max_steps = 100, random positions, random_walls = True"  # Description for the readme file
#####################################################


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


algorithm_name = algorithm.__name__
logdir, models_dir, model_name = set_scaffolding(
    env_name=env_name, algorithm_name=algorithm_name, policy=policy, device=device
)
t = int(time.time())


policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

env = make_shooting_minigrid_env(env_version=env_name, max_steps=max_steps, size=grid_size, random_walls=random_walls)
env = ImgObsWrapper(env)
env.reset()

# Training
model = algorithm(policy, env, verbose=1, policy_kwargs=policy_kwargs, device=device, tensorboard_log=logdir)

for i in range(1, number_of_saves + 1):
    model.learn(total_timesteps=timesteps_per_save, reset_num_timesteps=False, tb_log_name=f"{model_name}")
    model.save(f"{models_dir}/{timesteps_per_save * i}")
env.close()

write_info_file(
    env_name=env_name, algorithm_name=algorithm_name, model_name=model_name, policy=policy, device=device,
    episodes=number_of_saves * timesteps_per_save, models_dir=models_dir, description=description,
    time_elapsed=int(time.time()) - t
)

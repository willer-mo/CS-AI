import minigrid
from minigrid.wrappers import ImgObsWrapper
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import os
import time
from cs2d.d_custom_minigrid.custom_minigrid_env import CustomMinigridEnv



import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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




t = int(time.time())
algorithm = PPO
env_name = "CustomMinigrid-v1-size5"
# Device: cpu or cuda
device = "cpu"
algorithm_name = f"{algorithm.__name__}-{t}"
models_dir = f"../models/{env_name}/{algorithm_name}"
logdir = f"../logs/{env_name}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

env = CustomMinigridEnv(max_steps=5, size=5)
env = ImgObsWrapper(env)
env.reset()

# Training
# model = algorithm('MlpPolicy', env, verbose=1, device="cpu", tensorboard_log=logdir)
model = algorithm('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs, device=device, tensorboard_log=logdir)

TIMESTEPS = 10000
for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"{algorithm_name}")
    model.save(f"{models_dir}/{TIMESTEPS * i}")
env.close()


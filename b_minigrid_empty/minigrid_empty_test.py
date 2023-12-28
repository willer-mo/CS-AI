import minigrid
from minigrid.wrappers import ImgObsWrapper
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import os
import time



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
algorithm_name = f"{algorithm.__name__}-{t}"
models_dir = f"models/{algorithm_name}"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

env = gym.make('MiniGrid-Empty-16x16-v0', max_episode_steps=40, render_mode="human")
env.metadata["render_fps"] = 30
env = ImgObsWrapper(env)
env.reset()

episodes = 10

for ep in range(episodes):
    print(f"**************{ep}")
    observation, info = env.reset()
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if int(reward) != 0:
            print(reward)
        if terminated:
            print("Terminated!!")
env.close()


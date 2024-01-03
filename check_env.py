from stable_baselines3.common.env_checker import check_env
from c_custom_env.custom_env import CustomEnv
from d_shooting_minigrid.make_shooting_minigrid_env import make_shooting_minigrid_env
import gymnasium as gym

# Choose an env to check
#env = gym.make('LunarLander-v2')
#env = CustomEnv()
#env = gym.make('MiniGrid-Empty-16x16-v0')
env = make_shooting_minigrid_env(env_version="ShootingMiniGrid-v3", max_steps=50, size=25)

# It will check your custom environment and output additional warnings if needed
check_env(env)

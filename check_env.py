from stable_baselines3.common.env_checker import check_env
from cs2d.b_custom_env.custom_env import CustomEnv
from cs2d.d_custom_minigrid.custom_minigrid_env import CustomMinigridEnv
import gymnasium as gym

# Choose an env to check
#env = gym.make('LunarLander-v2')
#env = CustomEnv()
#env = gym.make('MiniGrid-Empty-16x16-v0')
env = CustomMinigridEnv()

# It will check your custom environment and output additional warnings if needed
check_env(env)

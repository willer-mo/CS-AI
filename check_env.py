from stable_baselines3.common.env_checker import check_env
from cs2d.c_custom_env.custom_env import CustomEnv
from cs2d.d_shooting_minigrid.shooting_minigrid_env import ShootingMiniGridEnv
import gymnasium as gym

# Choose an env to check
#env = gym.make('LunarLander-v2')
#env = CustomEnv()
#env = gym.make('MiniGrid-Empty-16x16-v0')
env = ShootingMiniGridEnv(render_mode="human", max_steps=5, size=5)

# It will check your custom environment and output additional warnings if needed
check_env(env)

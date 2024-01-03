from c_custom_env.custom_env import CustomEnv
from d_shooting_minigrid.make_shooting_minigrid_env import make_shooting_minigrid_env


# Choose an env to check
#env = gym.make('LunarLander-v2')
#env = CustomEnv()
#env = gym.make('MiniGrid-Empty-16x16-v0')
env = make_shooting_minigrid_env(env_version="ShootingMiniGrid-v2", render_mode="human", max_steps=5, size=25)

episodes = 50

for episode in range(episodes):
	print("*****************************************")
	terminated = False
	truncated = False
	env.reset()
	while not (terminated or truncated):
		random_action = env.action_space.sample()
		observation, reward, terminated, truncated, info = env.step(random_action)
		print("Action: ", random_action, "Reward: ", reward)

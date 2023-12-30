from cs2d.c_custom_env.custom_env import CustomEnv
from cs2d.d_shooting_minigrid.shooting_minigrid_env import ShootingMiniGridEnv


# Choose an env to check
#env = gym.make('LunarLander-v2')
#env = CustomEnv()
#env = gym.make('MiniGrid-Empty-16x16-v0')
env = ShootingMiniGridEnv(render_mode="human", max_steps=5, size=25, multi_action=True)

episodes = 50

for episode in range(episodes):
	print("*****************************************")
	terminated = False
	observation = env.reset()
	while not terminated:
		random_action = env.action_space.sample()
		print("action", random_action)
		observation, reward, terminated, truncated, info = env.step(random_action)
		if round(random_action[0]) == 3:
			print("SHOOTING!")
		#print('reward', reward, terminated)


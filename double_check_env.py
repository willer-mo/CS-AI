from cs2d.b_custom_env.custom_env import CustomEnv
from cs2d.d_custom_minigrid.custom_minigrid_env import CustomMinigridEnv


# Choose an env to check
#env = gym.make('LunarLander-v2')
#env = CustomEnv()
#env = gym.make('MiniGrid-Empty-16x16-v0')
env = CustomMinigridEnv()

episodes = 50

for episode in range(episodes):
	terminated = False
	observation = env.reset()
	while not terminated:
		random_action = env.action_space.sample()

		print("*****************************************")
		print(f"observation\n{observation[0:5]}\n{observation[5:10]}\n{observation[10:15]}\n{observation[15:20]}\n{observation[20:25]}")
		print("action", random_action)
		observation, reward, terminated, truncated, info = env.step(random_action)
		print('reward', reward, terminated)


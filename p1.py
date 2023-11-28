import gymnasium as gym
from stable_baselines3 import PPO
import time

env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2
env.reset()

# Learn
t1 = time.time()
model = PPO('MlpPolicy', env, verbose=1, device="cpu")
model.learn(total_timesteps=4000)
print(f"Time training with cuda: {time.time()-t1:.2f}s")
env.close()

# Test
# env = gym.make('LunarLander-v2', render_mode="human")  # continuous: LunarLanderContinuous-v2
# env.metadata["render_fps"] = 240
# episodes = 5
#
# for ep in range(episodes):
# 	# print(ep)
# 	observation, info = env.reset()
# 	terminated = False
# 	while not terminated:
# 		action, _states = model.predict(observation)
# 		observation, reward, terminated, truncated, info = env.step(action)
# 		env.render()
# 		print(reward)
# env.close()


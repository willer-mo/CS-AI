import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import os
import time

t = int(time.time())
algorithm = PPO
env_name = "LunarLander-v2"
# Device: cpu or cuda
device = "cpu"
algorithm_name = f"{algorithm.__name__}-{t}"
models_dir = f"../models/{env_name}/{algorithm_name}"
logdir = f"../logs/{env_name}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make(env_name)
env.reset()

# Training
model = algorithm('MlpPolicy', env, verbose=1, device=device, tensorboard_log=logdir)
TIMESTEPS = 10000
for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"{algorithm_name}")
    model.save(f"{models_dir}/{TIMESTEPS * i}")
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


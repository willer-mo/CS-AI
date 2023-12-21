from pcustom import CustomEnv
import os
import time
from stable_baselines3 import PPO

t = int(time.time())
algorithm = PPO
model_name = f"{algorithm.__name__}-custom5x5-{t}"
models_dir = f"models/{model_name}"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


env = CustomEnv(max_episode_steps=10)
env.reset()

# Training
model = algorithm('MlpPolicy', env, verbose=1, device="cpu", tensorboard_log=logdir)
# model = algorithm('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs, device="cpu", tensorboard_log=logdir)
TIMESTEPS = 10000
for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"{model_name}")
    model.save(f"{models_dir}/{TIMESTEPS * i}")
env.close()


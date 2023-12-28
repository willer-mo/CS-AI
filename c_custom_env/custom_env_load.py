from stable_baselines3 import PPO
from stable_baselines3 import A2C
from custom_env import CustomEnv
import os

################## USER PARAMETERS ##################
env_name = "CustomEnv_5x5"
algorithm = PPO
suffix = "2"  # Sufijo del modelo guardado en la carpeta models
zip_model = "290000.zip"  # Carpeta zip del modelo
episodes = 5
#####################################################

model_name = f"{algorithm.__name__}{'_' + suffix if suffix else ''}"
models_dir = f"../models/{env_name}/{model_name}"
model_path = f"{models_dir}/{zip_model}"

env = CustomEnv(render_mode="human")
env.reset()

# LOAD and RUN
model = algorithm.load(model_path, env=env)

for ep in range(episodes):
    observation, info = env.reset()
    env.render()
    terminated = False
    while not terminated:
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward}")
        if terminated:
            print("Terminated!!")
env.close()

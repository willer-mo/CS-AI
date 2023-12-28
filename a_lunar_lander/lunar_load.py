import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C


################## USER PARAMETERS ##################
env_name = "LunarLander-v2"
algorithm = PPO
suffix = "1"  # Sufijo del modelo guardado en la carpeta models
zip_model = "290000.zip"  # Carpeta zip del modelo
episodes = 15
#####################################################

model_name = f"{algorithm.__name__}{'_' + suffix if suffix else ''}"
models_dir = f"../models/{env_name}/{model_name}"
model_path = f"{models_dir}/{zip_model}"

env = gym.make(env_name, render_mode="human", max_episode_steps=500)
env.metadata["render_fps"] = 500

# LOAD and RUN
model = algorithm.load(model_path, env=env)

for ep in range(episodes):
    observation, info = env.reset()
    terminated = False
    while not terminated:
        action, _states = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print("Terminated!!")
        env.render()
env.close()

from gymnasium.envs.registration import register

register(
    id="CS2D-v0",
    entry_point="CS2D.envs:CS2DEnv",
    max_episode_steps=300,
)

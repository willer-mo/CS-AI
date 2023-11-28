from gymnasium.envs.registration import register

register(
    id="CS2D-v0",
    entry_point="gym_game.envs:CS2DEnv",
    max_episode_steps=300,
)

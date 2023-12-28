import os
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, max_episode_steps=100, render_mode=None, map_size=(5, 5)):
        super(CustomEnv, self).__init__()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.map_size = map_size
        self.observation_space = spaces.Box(low=0, high=2, shape=(map_size[0] * map_size[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 4 possible actions: up, down, left, right

        self.agent_position = (0, 0)
        self.goal_position = (np.random.randint(0, map_size[0]), np.random.randint(0, map_size[1]))

    def step(self, action):
        # Update agent position based on the action
        if action == 0:  # Up
            self.agent_position = (max(0, self.agent_position[0] - 1), self.agent_position[1])
        elif action == 1:  # Down
            self.agent_position = (min(self.map_size[0] - 1, self.agent_position[0] + 1), self.agent_position[1])
        elif action == 2:  # Left
            self.agent_position = (self.agent_position[0], max(0, self.agent_position[1] - 1))
        elif action == 3:  # Right
            self.agent_position = (self.agent_position[0], min(self.map_size[1] - 1, self.agent_position[1] + 1))

        # Check if the agent has reached the goal
        terminated = (self.agent_position == self.goal_position)

        # Calculate reward (you can define your own reward function)
        reward = 1.0 if terminated else 0.0

        # Return the next observation, reward, and done flag
        observation = self._get_observation()
        info = {}
        truncated = False
        self.current_step = self.current_step + 1
        if self.current_step == self.max_episode_steps and not terminated:
            terminated = True
            truncated = True
            reward = 0

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        # Generate an observation representing the agent's position and the goal position
        observation = np.zeros(self.map_size, dtype=np.float32)
        observation[self.agent_position] = 1.0  # Agent position
        observation[self.goal_position] = 2.0  # Goal position
        return self.flatten_extend(observation)

    @staticmethod
    def flatten_extend(matrix):
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return np.array(flat_list)

    def reset(self, seed=None, options=None):
        # Reset agent position and goal position
        self.agent_position = (0, 0)
        goal_position = (0, 0)
        while goal_position == (0, 0):
            goal_position = (
                np.random.randint(0, self.map_size[0]),
                np.random.randint(0, self.map_size[1])
            )
        self.goal_position = goal_position
        self.current_step = 0

        # Return initial observation
        observation = self._get_observation()
        info = {}
        return observation, info

    def render(self):
        if self.render_mode == "human":
            observation = self._get_observation()
            time.sleep(1)
            os.system('cls' if os.name == 'nt' else 'clear')
            print(
                f"\n{observation[0:5]}\n{observation[5:10]}\n{observation[10:15]}\n{observation[15:20]}\n{observation[20:25]}")

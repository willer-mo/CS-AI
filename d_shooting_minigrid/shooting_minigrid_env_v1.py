from __future__ import annotations
from typing import Any, Iterable, SupportsFloat, TypeVar
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from cs2d.d_shooting_minigrid.shooting_minigrid_env import ShootingMiniGridBaseEnv


class ShootingMiniGridEnvV1(ShootingMiniGridBaseEnv):
    def __init__(
        self,
        multi_action=False,
        **kwargs,
    ):
        self.multi_action = multi_action
        super().__init__(**kwargs)

        ### Action space ###
        # Define discrete actions for each type of action: action_space[0]
        # 0: do nothing
        # 1: movement
        # 2: rotation
        # 3: shooting

        # Define discrete actions for movement: action_space[1]
        # 0: up
        # 1: right
        # 2: down
        # 3: left

        # Define continuous actions for changing direction: action_space[2]
        # Range [-1, 1] of radians. Equivalent to [-57, 57] degrees approx.

        self.action_space = gym.spaces.Box(low=np.array([0, 0, -1]), high=np.array([3, 3, 1]), dtype=np.float32)

        # # Uncomment for only direction and shooting (no movement).
        # self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        # Observation space: The same as in MiniGridEnv but with continuous direction
        # Removed mission keyword for incompatibilities when executing stable_baselines3.common.env_checker
        image = self.observation_space.get("image")
        # Image definition in MiniGridEnv is a 3 dimension matrix:
        # spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(self.agent_view_size, self.agent_view_size, 3),
        #     dtype="uint8",
        # )
        self.observation_space = spaces.Dict(
            {
                "image": image,
                "direction": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64),
            }
        )

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        discrete_action, movement, direction = action
        if discrete_action:
            discrete_action = round(discrete_action)
        if movement:
            movement = round(movement)

        # # Uncomment for only direction and shooting (no movement).
        # direction, shooting = action
        # movement = False
        # discrete_action = 3

        if isinstance(direction, tuple):
            direction = self.calculate_agents_angle(direction)
        reward = 0
        terminated = False
        truncated = False

        next_move_pos = False
        multi_action = self.multi_action

        if discrete_action == 0 or multi_action:
            # Do nothing
            pass

        # Moving action
        if discrete_action == 1 or multi_action and movement is not False:
            # Move left
            if movement == 3:
                next_move_pos = self.left_pos
            # Move right
            elif movement == 1:
                next_move_pos = self.right_pos
            # Move forward
            elif movement == 0:
                next_move_pos = self.front_pos
            # Move backwards
            elif movement == 2:
                next_move_pos = self.back_pos

        # Rotation
        if discrete_action == 2 or multi_action and direction is not False:
            self.update_agents_rotation(direction)

        # Shooting
        if discrete_action == 3:  # and shooting > 0.0:  # Uncomment for no movement.
            hit = self.shooting()
            if hit:
                terminated = True
                reward = self._reward()

        if next_move_pos is not False:
            # Get the contents of the next cell the agent wants to move to
            next_move_cell = self.grid.get(*next_move_pos)
            if next_move_cell is None or next_move_cell.can_overlap():
                self.agent_pos = tuple(next_move_pos)

        if self.step_count >= self.max_steps:
            truncated = True
        if self.render_mode == "human":
            self.render()
        obs = self.gen_obs()
        return obs, reward, terminated, truncated, {}

    def gen_obs(self):
        obs = super().gen_obs()
        # Establecer la dirección contínua en formato float
        obs.update({"direction": np.array([self.agent_dir])})
        # Removed mission keyword for incompatibilities when executing stable_baselines3.common.env_checker
        obs.pop("mission")
        return obs

from __future__ import annotations
from typing import Any
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import OBJECT_TO_IDX

from cs2d.d_shooting_minigrid.shooting_minigrid_env_v2 import ShootingMiniGridEnvV2


class ShootingMiniGridEnvV3(ShootingMiniGridEnvV2):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Same action space as in ShootingMiniGridEnvV2
        # 0-90: degrees of rotation. 0=-45º, 45=0º, 90=45º
        # 91: move forward
        # 92: move left
        # 93: move backwards
        # 94: move right
        # 95: shoot

        # Observation space: 1 dimensional array of the size of the grid with an extra value for the agent's direction
        # All values of the array range within [0, 10] except the last one that ranges within [-pi, pi]
        view_size = self.agent_view_size * self.agent_view_size + 1
        self.observation_space = spaces.Box(
            low=np.append(np.full(view_size - 1, 0), -180),
            high=np.append(np.full(view_size - 1, 10), 180),
            shape=(view_size,),
            dtype="int16",
        )
        # self.agent_target_distance = 0
        # self.agent_target_angle_diff = 0

    # def reset(
    #         self,
    #         *,
    #         seed: int | None = None,
    #         options: dict[str, Any] | None = None,
    # ) -> tuple[ObsType, dict[str, Any]]:
    #     obs, info = super().reset(seed=seed)
    #     self.agent_target_distance = self.get_agent_target_distance()
    #     self.agent_target_angle_diff = self.get_agent_target_angle_diff()
    #     return obs, info

    # def get_agent_target_distance(self):
    #     return np.linalg.norm(np.array(self.agent_pos) - np.array(self.target_position))
    #
    # def get_agent_target_angle_diff(self):
    #     # Calculate the vector from the agent to the object
    #     target_vector = np.array(self.target_position) - np.array(self.agent_pos)
    #     # Calculate the angle between the agent's direction and the object vector
    #     angle_difference = -np.arctan2(target_vector[1], target_vector[0]) - self.agent_dir
    #     return round(abs(angle_difference), 2)

    def step(self, action):
        # Same step as in ShootingMiniGridEnvV2
        return super().step(action)

    # def step(self, action):
    #     # Same step as in ShootingMiniGridEnvV2
    #     obs, reward, terminated, truncated, info = super().step(action)
    #     if not terminated:
    #         if action in [91, 92, 93, 94]:
    #             # Moving action
    #             previous_distance = self.agent_target_distance
    #             current_distance = self.get_agent_target_distance()
    #             distance_diff = previous_distance - current_distance
    #             if current_distance > 7:
    #                 # Si se encuentra a menos de 5 celdas de distancia, no premiar el movimiento.
    #                 if distance_diff > 0:
    #                     reward = 0.1
    #                 elif distance_diff < 0:
    #                     reward = - 0.1
    #             self.agent_target_distance = current_distance
    #             self.agent_target_angle_diff = self.get_agent_target_angle_diff()
    #
    #         elif 0 <= action <= 90:
    #             # Rotation
    #             previous_angle = self.agent_target_angle_diff
    #             current_angle = self.get_agent_target_angle_diff()
    #             angle_diff = previous_angle - current_angle
    #             if angle_diff > 0:
    #                 reward = 0.1
    #             elif angle_diff < 0:
    #                 reward = - 0.1
    #             self.agent_target_angle_diff = current_angle
    #     return obs, reward, terminated, truncated, info

    # def _reward(self) -> float:
    #     """
    #     Compute the reward to be given upon success
    #     """
    #
    #     return (1 - 0.9 * (self.step_count / self.max_steps)) * 100

    def gen_obs(self):
        grid = self.grid

        # Encode the grid view into a numpy array
        obs = self.encode(grid)

        obs[self.agent_pos[0]][self.agent_pos[1]] = OBJECT_TO_IDX["agent"]

        obs = self.flatten_extend(obs)

        # Add the agent's direction at the end
        direction = round(np.degrees(self.agent_dir))
        # Normalize the new direction to be between -180 and 180
        # dir = (dir + 180) % (360) - 180
        obs = np.append(obs, np.array([direction], dtype="int16"))
        return obs

    @staticmethod
    def flatten_extend(matrix):
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return np.array(flat_list)

from __future__ import annotations
from typing import Any
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
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

    def step(self, action):
        # Same step as in ShootingMiniGridEnvV2
        return super().step(action)

    def gen_obs(self):
        grid = self.grid

        # Encode the grid view into a numpy array
        obs = self.encode(grid)
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

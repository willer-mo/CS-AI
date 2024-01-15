from __future__ import annotations
from typing import Any, Iterable, SupportsFloat, TypeVar
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from minigrid.core.constants import OBJECT_TO_IDX, TILE_PIXELS
from cs2d.d_shooting_minigrid.shooting_minigrid_env import ShootingMiniGridBaseEnv


class ShootingMiniGridEnvV2(ShootingMiniGridBaseEnv):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Define discrete actions for each type of action
        # 0-90: degrees of rotation. 0=-45º, 45=0º, 90=45º
        # 91: move forward
        # 92: move left
        # 93: move backwards
        # 94: move right
        # 95: shoot
        self.action_space = spaces.Discrete(96)

        # Observation space: The same as in ShootingMiniGridEnvV1 but an image of 2 dimensions instead of 3.
        image = spaces.Box(
            low=0,
            high=10,
            shape=(self.agent_view_size, self.agent_view_size),
            dtype="uint8",
        )
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

        reward = 0
        terminated = False
        truncated = False

        next_move_pos = False

        if 0 <= action <= 90:
            # Rotation
            direction = np.radians(action - 45)
            self.update_agents_rotation(direction)

        elif action in [91, 92, 93, 94]:
            # Moving action

            if action == 91:
                # Move forward
                next_move_pos = self.front_pos
            elif action == 92:
                # Move left
                next_move_pos = self.left_pos
            elif action == 93:
                # Move backwards
                next_move_pos = self.back_pos
            elif action == 94:
                # Move right
                next_move_pos = self.right_pos

        elif action == 95:
            # Shooting
            hit = self.shooting()
            if hit:
                terminated = True
                reward = self._reward()

        else:
            raise ValueError(f"Unknown action: {action}")

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
        grid, vis_mask = self.gen_obs_grid()

        # Encode the fully observable view into a numpy array
        image = self.encode(grid)
        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        obs = {"image": image, "direction": np.array([self.agent_dir])}
        return obs

    def encode(self, grid) -> np.ndarray:
        """
        Produce a compact numpy encoding of the grid
        Same as MiniGridEnv but with a 2 dimension array instead of 3.
        """
        array = np.zeros((self.width, self.height), dtype="uint8")

        for i in range(self.width):
            for j in range(self.height):
                v = grid.get(i, j)
                if v is None:
                    array[i, j] = OBJECT_TO_IDX["empty"]
                else:
                    array[i, j] = OBJECT_TO_IDX[v.type]
        return array

    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """
        grid = self.grid

        # This vis_mask indicates that the agent can see the whole grid, so there is no partial observation.
        vis_mask = np.ones((self.width, self.height), dtype=bool)

        return grid, vis_mask

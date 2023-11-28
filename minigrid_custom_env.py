from __future__ import annotations
from actions_cs import ActionsCS

from typing import Any, Iterable, SupportsFloat, TypeVar

from gymnasium.core import ActType, ObsType

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MinigridCustomEnv(MiniGridEnv):
    # metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(
        self,
        size=25,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size ** 2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            agent_view_size=11,
            **kwargs,
        )


    @staticmethod
    def _gen_mission():
        return "get to the green cell"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate verical separation wall
        for i in range(0, height):
            if i in [5, 6, 7, 8, 9, 14, 15, 16, 17, 18]:
                continue
            self.grid.set(5, i, Wall())

        for i in range(0, height):
            if i in [5, 6, 7, 8, 9, 14, 15, 16, 17, 18]:
                continue
            self.grid.set(19, i, Wall())

        self.grid.set(12, 6, Wall())
        self.grid.set(12, 7, Wall())
        self.grid.set(12, 8, Wall())
        self.grid.set(12, 15, Wall())
        self.grid.set(12, 16, Wall())
        self.grid.set(12, 17, Wall())
        # Place the door and key
        # self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        # self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - int(height/2))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = self._gen_mission()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Move left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Move right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    # def _get_observation(self):
    #     # Generate an observation representing the agent's position and the goal position
    #     observation = np.zeros(self.map_size, dtype=np.float32)
    #     observation[self.agent_position] = 1.0  # Agent position
    #     observation[self.goal_position] = 2.0  # Goal position
    #     return self.flatten_extend(observation)
    #
    # @staticmethod
    # def flatten_extend(matrix):
    #     flat_list = []
    #     for row in matrix:
    #         flat_list.extend(row)
    #     return np.array(flat_list)
    #
    # def reset(self, seed=None, options=None):
    #     # Reset agent position and goal position
    #     self.agent_position = (0, 0)
    #     goal_position = (0, 0)
    #     while goal_position == (0, 0):
    #         goal_position = (
    #             np.random.randint(0, self.map_size[0]),
    #             np.random.randint(0, self.map_size[1])
    #         )
    #     self.goal_position = goal_position
    #     self.current_step = 0
    #
    #     # Return initial observation
    #     observation = self._get_observation()
    #     info = {}
    #     return observation, info
    #
    # def render(self):
    #     if self.render_mode == "human":
    #         observation = self._get_observation()
    #         print(
    #             f"****************\n{observation[0:5]}\n{observation[5:10]}\n{observation[10:15]}\n{observation[15:20]}\n{observation[20:25]}")

    # def close(self):
    #     ...


def main():
    env = MinigridCustomEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import Any, Iterable, SupportsFloat, TypeVar

from gymnasium.core import ActType, ObsType

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
import random
import pygame

import gymnasium as gym
from gymnasium import spaces
import numpy as np

#from world_object import WallExtended


class ShootingMiniGridEnv(MiniGridEnv):
    # metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(
        self,
        size=25,
        agent_start_dir=0.0,
        max_steps: int | None = None,
        multi_action=False,
        moving_speed=1,
        **kwargs,
    ):
        self.agent_start_pos = 1, random.randint(1, size - 2)
        self.agent_start_dir = agent_start_dir
        self.multi_action = multi_action
        self.moving_speed = moving_speed

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size ** 2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            agent_view_size=size,
            **kwargs,
        )
        #self.agent_pov = True

        # # Define discrete actions for each type of action
        # # 0: do nothing
        # # 1: movement
        # # 2: rotation
        # # 3: shooting
        # self.discrete_action = spaces.Discrete(4)
        #
        # # Define discrete actions for movement
        # # 0: up
        # # 1: right
        # # 2: down
        # # 3: left
        # self.movement_action_space = spaces.Discrete(4)
        #
        # # Define continuous actions for changing direction (360 degrees)
        # self.direction_action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # # self.shoot_action_space = spaces.Discrete(2)  # 0 for not shooting, 1 for shooting
        #
        # # Combine discrete and continuous action spaces
        # self.action_space = spaces.Tuple([
        #     self.discrete_action, self.movement_action_space, self.direction_action_space
        # ])
        # self.action_space = spaces.Dict({
        #     'discrete_action': self.discrete_action,
        #     'movement': self.discrete_action,
        #     'direction': self.direction_action_space,  # Continuous rotation
        # })
        self.action_space = gym.spaces.Box(low=np.array([0, 0, -1]), high=np.array([3, 3, 1]), dtype=np.float32)

        # Observation space: The same as in MiniGridEnv but with continuous direction
        image = self.observation_space.get("image")
        self.observation_space = spaces.Dict(
            {
                "image": image,
                "direction": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64),
                #"mission": mission_space,
            }
        )

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        dx = round(np.cos(self.agent_dir))
        dy = -round(np.sin(self.agent_dir))
        return np.array((dx, dy))

    @property
    def left_vec(self):
        """
        Get the vector pointing to the left of the agent.
        """
        left_vector = -self.right_vec
        return left_vector

    @property
    def left_pos(self):
        """
        Get the position of the cell that is left of the agent
        """

        return self.agent_pos + self.left_vec

    @property
    def right_pos(self):
        """
        Get the position of the cell that is left of the agent
        """

        return self.agent_pos + self.right_vec

    @property
    def back_pos(self):
        """
        Get the position of the cell that is behind of the agent
        """
        back_vector = -self.dir_vec
        return self.agent_pos + back_vector

    @staticmethod
    def _gen_mission():
        return "Shoot the green cell"

    # def gen_obs_grid(self, agent_view_size=None):
    #  return

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # # Generate verical separation wall
        # for i in range(0, height):
        #     if i in [5, 6, 7, 8, 9, 14, 15, 16, 17, 18]:
        #         continue
        #     #self.grid.set(5, i, WallExtended())
        #     self.grid.set(5, i, Wall())
        #
        # for i in range(0, height):
        #     if i in [5, 6, 7, 8, 9, 14, 15, 16, 17, 18]:
        #         continue
        #     self.grid.set(19, i, Wall())
        #
        # self.grid.set(12, 6, Wall())
        # self.grid.set(12, 7, Wall())
        # self.grid.set(12, 8, Wall())
        # self.grid.set(12, 15, Wall())
        # self.grid.set(12, 16, Wall())
        # self.grid.set(12, 17, Wall())

        # Place a goal square ) in a random cell of the last column of the grid
        target_position = width - 2, random.randint(1, height-2)
        self.put_obj(Goal(), target_position[0], target_position[1])
        self.target_position = target_position

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

        discrete_action, movement, direction = action
        if discrete_action:
            discrete_action = round(discrete_action)
        if movement:
            movement = round(movement)
        #discrete_action, movement, direction = (action["discrete_action"], action["movement"], action["direction"][0])
        if isinstance(direction, tuple):
            direction = self.calculate_agents_angle(direction)
        reward = 0
        terminated = False
        truncated = False

        next_move_pos = False
        multi_action = self.multi_action

        if discrete_action == 0 or multi_action:
            # do nothing
            pass

        # Moving action
        if discrete_action == 1 or multi_action and movement is not False:
            # Move left
            #if action == self.actions.left:
            if movement == 3:
                next_move_pos = self.left_pos
            # Move right
            #elif action == self.actions.right:
            elif movement == 1:
                next_move_pos = self.right_pos
            # Move forward
            #elif action == self.actions.forward:
            elif movement == 0:
                next_move_pos = self.front_pos
            #elif action == self.actions.backward:
            elif movement == 2:
                next_move_pos = self.back_pos

        # Rotation
        if discrete_action == 2 or multi_action and direction is not False:
            self.update_agents_rotation(direction)

        # Shooting
        if discrete_action == 3:
            hit = self.shooting()
            if hit:
                terminated = True
                reward = self._reward()
        # else:
        #     raise ValueError(f"Unknown action: {action}")

        if next_move_pos is not False:
            # Get the contents of the next cell the agent wants to move to
            next_move_cell = self.grid.get(*next_move_pos)
            if next_move_cell is None or next_move_cell.can_overlap():
                self.agent_pos = tuple(next_move_pos)
            # if next_move_cell is not None and next_move_cell.type == "goal":
            #     terminated = True
            #     reward = self._reward()
            # if next_move_cell is not None and next_move_cell.type == "lava":
            #     terminated = True

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
        #obs.update({"mission": np.array([self.mission])})
        obs.pop("mission")
        return obs

    def shooting(self, allowed_deviation=np.radians(2)):
        # returns whether the agent hits the enemy or not
        agent_position = self.agent_pos
        agent_angle = self.agent_dir
        # Calculate the vector from the agent to the object
        target_vector = np.array(self.target_position) - np.array(agent_position)

        # Calculate the angle between the agent's direction and the object vector
        angle_difference = -np.arctan2(target_vector[1], target_vector[0]) - agent_angle

        # Normalize the angle to be between -pi and pi
        angle_difference = (angle_difference + np.pi) % (2 * np.pi) - np.pi
        #print(f"Target: {np.degrees(-np.arctan2(object_vector[1], object_vector[0]))} Agent: {np.degrees(self.agent_dir)}")
        # Check if the absolute angle difference is within the allowed deviation
        return abs(angle_difference) <= allowed_deviation

    def update_agents_rotation(self, direction):
        self.agent_dir += direction

    def get_agents_fov(self):
        x, y = self.agent_pos
        fov_angle = np.radians(90)  # 90 degrees of Field of View (fov)
        direction = self.agent_dir
        # Initialize a matrix to represent the grid (all zeros)
        mask = np.zeros((self.width, self.height), dtype=bool)

        # Highlight cells only in front of the player
        mask[y, x] = True

        for row in range(self.height):
            for col in range(self.width):
                dx = col - x
                dy = row - y
                angle_to_cell = np.arctan2(dy, dx) + direction
                angle_to_cell = (angle_to_cell + np.pi) % (2 * np.pi) - np.pi

                if -fov_angle / 2 <= angle_to_cell <= fov_angle / 2:
                    mask[row, col] = 1
        return mask

    def render(self):
        img = super().render()
        if self.render_mode == "human":
            x, y = self.get_agents_screen_pos()
            # Calculate the end point at the edge of the screen
            end_x, end_y = self.get_end_aiming_point((x, y))
            pygame.draw.line(self.window, (252, 74, 74), (x, y), (end_x, end_y), 2)
            # Draw agent
            offset = 60
            radius = ((self.screen_size - offset) / min(self.width, self.height)) / 2
            pygame.draw.circle(self.window, (252, 74, 74), (x, y), radius)
            pygame.display.flip()
        return img

    def get_end_aiming_point(self, agent_screen_pos):
        angle = self.agent_dir
        d_angle = np.degrees(self.agent_dir)
        start_x, start_y = agent_screen_pos
        start_y *= -1
        offset = 60
        start_x -= int(offset/2)
        height = width = self.screen_size - offset
        with np.errstate(divide='ignore'):
            if d_angle >= 0 and d_angle < 90:
                length = min(
                    (width - start_x) / np.cos(angle),
                    (height - start_y) / np.sin(angle)
                )
            elif d_angle >= 90 and d_angle < 180:
                length = min(
                    (start_x) / np.cos(np.pi - angle),
                    (height - start_y) / np.sin(np.pi - angle))
            elif d_angle >= -180 and d_angle <= -90:
                length = min(
                    (start_x) / np.cos(angle + np.pi),
                    (height + start_y) / np.sin(angle + np.pi))
            else:
                length = min(
                    (width - start_x) / np.cos(angle + 2 * np.pi),
                    -(height + start_y) / np.sin(angle + 2 * np.pi))
        if np.isnan(length):
            length = 0
        end_x = round(start_x + length * np.cos(angle)) + offset / 2
        end_y = -round(start_y + length * np.sin(angle))
        return end_x, end_y

    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        """
        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=vis_mask.T if highlight else None,
        )

        return img

    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """
        grid = self.grid
        agent_view_size = agent_view_size or self.agent_view_size
        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size - 1)
            )
        else:
            vis_mask = self.get_agents_fov()

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        return grid, vis_mask

    def calculate_agents_angle(self, aim_pos):
        # Function used only in manual mode
        a, b = self.agent_pos
        x, y = aim_pos
        current_angle = self.agent_dir
        new_angle = -np.arctan2(y - b, x - a)
        return new_angle - current_angle

    def get_agents_screen_pos(self):
        # Function used only in human render
        offset = 60
        cell_x, cell_y = self.agent_pos
        # Get the screen position of the center of the cell
        cell_x += 0.5
        cell_y += 0.5

        pixel_per_cell_x = (self.screen_size - offset) / self.width
        pixel_per_cell_y = (self.screen_size - offset) / self.height
        # Calculate pos
        pos_x = round((cell_x * pixel_per_cell_x) + (offset / 2))
        pos_y = round((cell_y * pixel_per_cell_y))
        return pos_x, pos_y

from __future__ import annotations
import random
import pygame
import numpy as np
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv


class ShootingMiniGridBaseEnv(MiniGridEnv):
    def __init__(
        self,
        size=25,
        agent_start_dir=0.0,
        max_steps: int | None = None,
        moving_speed=1,
        agent_start_pos: tuple | None = None,
        target_position: tuple | None = None,
        see_through_walls=True,
        random_walls=False,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.random_agent_start_pos = False
        if not agent_start_pos:
            self.random_agent_start_pos = True

        self.target_position = target_position
        self.random_target_position = False
        if not target_position:
            self.random_target_position = True

        self.agent_start_dir = agent_start_dir

        self.moving_speed = moving_speed
        self.random_walls = random_walls

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size ** 2

        self.size = size

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=see_through_walls,
            max_steps=max_steps,
            agent_view_size=size,
            **kwargs,
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

    def add_random_walls(self, width, height):
        # Generate verical separation wall
        number_of_walls = round(height / 2)
        x_walls = [5, round(width / 2), width - 6]
        for x in x_walls:
            y_walls = random.sample(range(1, height - 1), number_of_walls)
            for y in y_walls:
                self.grid.set(x, y, Wall())

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        if self.random_walls:
            self.add_random_walls(width, height)

        if self.random_agent_start_pos:
            self.agent_start_pos = 1, random.randint(1, self.size - 2)

        if self.random_target_position:
            self.target_position = self.size - 2, random.randint(1, self.size - 2)

        self.put_obj(Goal(), self.target_position[0], self.target_position[1])

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = self._gen_mission()

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

        # Check if the absolute angle difference is within the allowed deviation
        hit = abs(angle_difference) <= allowed_deviation
        if hit:
            # Check if there is a wall in the way of the shot
            return self.check_for_wall_hit()
        return False

    def check_for_wall_hit(self):
        """
        Bresenham's Line Algorithm to check if any of the cells of the shot line are walls.
        Returns:
        - True if there are no walls between the agent and the target or False if there is.
        """
        start_x, start_y = self.agent_pos
        end_x, end_y = self.target_position

        dx = abs(end_x - start_x)
        dy = abs(end_y - start_y)

        sx = -1 if start_x > end_x else 1
        sy = -1 if start_y > end_y else 1

        err = dx - dy

        x, y = start_x, start_y

        while True:
            cell = self.grid.get(x, y)
            if cell is not None and cell.type == "wall":
                return False
            if x == end_x and y == end_y:
                break

            e2 = 2 * err

            if e2 > -dy:
                err = err - dy
                x = x + sx

            if e2 < dx:
                err = err + dx
                y = y + sy

        return True

    def update_agents_rotation(self, direction):
        new_dir = self.agent_dir + direction
        # Normalize the new direction to be between -pi and pi
        new_dir = (new_dir + np.pi) % (2 * np.pi) - np.pi
        self.agent_dir = new_dir

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

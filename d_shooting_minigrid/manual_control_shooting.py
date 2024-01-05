import pygame
import numpy as np
from minigrid.manual_control import ManualControl
from shooting_minigrid_env_v1 import ShootingMiniGridEnvV1


class ManualControlShooting(ManualControl):
    def __init__(self, env, seed=None):
        super().__init__(env, seed=seed)
        self.mouse_pos = False

    def key_handler(self, events):
        # super().key_handler(event)
        # action = [type of action, movement, rotation]
        key: str = events[1]
        action_type = events[0]
        mouse_pos = events[2]
        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        key_to_action = {
            "up": 0,
            "right": 1,
            "down": 2,
            "left": 3,
            "w": 0,
            "a": 3,
            "s": 2,
            "d": 1,
        }
        move_action = False
        if key in key_to_action.keys():
            move_action = key_to_action[key]
        if mouse_pos:
            self.mouse_pos = mouse_pos
            mouse_pos = self.convert_pos_to_cell(mouse_pos)
        action = tuple([action_type, move_action, mouse_pos])
        self.step(action)

    def convert_pos_to_cell(self, mouse_pos):
        # Calculate pixel per cell
        offset = 60
        pixel_per_cell_x = (self.env.screen_size - offset) / self.env.width
        pixel_per_cell_y = (self.env.screen_size - offset) / self.env.height
        # Calculate cell
        cell_x = np.floor((mouse_pos[0] - (offset / 2)) / pixel_per_cell_x)
        cell_y = np.floor(mouse_pos[1] / pixel_per_cell_y)
        return (cell_x, cell_y)

    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)

        while not self.closed:
            # events = [type of action, movement, rotation]
            events = [0, False, False]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                elif event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    events[1] = event.key
                    if event.key == "space":
                        events[0] = 3
                elif event.type == pygame.MOUSEMOTION:
                    events[2] = event.pos
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        events[0] = 3
                        events[2] = event.pos

            self.key_handler(events)

    def step(self, action):
        _, reward, terminated, truncated, _ = self.env.step(action)
        if terminated:
            print(f"step={self.env.step_count}, reward={reward:.2f}")
            print("terminated!")
            self.reset(self.seed)
        elif truncated:
            print(f"step={self.env.step_count}, reward={reward:.2f}")
            print("truncated!")
            self.reset(self.seed)


def main():
    env = ShootingMiniGridEnvV1(render_mode="human", multi_action=True, size=25, random_walls=True)

    # Enable manual control for testing
    manual_control = ManualControlShooting(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()

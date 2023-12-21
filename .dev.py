# Script for development and testing the render of the environments
import pygame
import sys
import math
import numpy as np

# Initialize Pygame
pygame.init()

# Set the grid size and cell size
grid_size = 50
cell_size = 20

# Calculate the window size
width, height = grid_size * cell_size, grid_size * cell_size

# Set the colors
background_color = (255, 255, 255)  # White
grid_color = (200, 200, 200)  # Light gray
player_color = (0, 0, 255)  # Blue
player_color2 = (0, 0, 0)  # Blue
highlight_color = (255, 0, 0)  # Red

# Set the player's initial position (cell center) and rotation angle
player_position = np.array([5, 25])
player_rotation = 0  # Initial rotation angle in degrees

# Create the Pygame window
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Grid with Player")

def draw_grid():
    # Draw the grid
    for row in range(grid_size):
        for col in range(grid_size):
            pygame.draw.rect(screen, grid_color, (col * cell_size, row * cell_size, cell_size, cell_size), 1)

def draw_player(position, rotation):
    # Draw the player as a triangle
    x, y = position
    # player_points = [(x * cell_size, y * cell_size),
    #                  ((x + 0.5) * cell_size, (y + 1) * cell_size),
    #                  (x * cell_size, (y + 1) * cell_size)]
    # rotated_player = pygame.transform.rotate(pygame.Surface((cell_size, cell_size), pygame.SRCALPHA), -rotation)
    # pygame.draw.polygon(rotated_player, player_color, player_points)
    # screen.blit(rotated_player, (x * cell_size - cell_size / 2, y * cell_size - cell_size / 2))
    pygame.draw.rect(screen, player_color, (x * cell_size, y * cell_size, cell_size, cell_size), 1)

def draw_highlighted_cells(position, rotation):
    x, y = position
    angle = math.radians(90)  # 90 degrees

    # Find the cell right behind the player

    # Initialize a matrix to represent the grid (all zeros)
    highlighted_matrix = np.zeros((grid_size, grid_size), dtype=int)

    # Highlight cells only in front of the player
    highlighted_matrix[x, y] = 1
    pygame.draw.rect(screen, highlight_color, (x * cell_size, y * cell_size, cell_size, cell_size))
    for row in range(grid_size):
        for col in range(grid_size):
            dx = col - x
            dy = row - y
            #angle_to_cell = math.atan2(dy, dx) + math.radians(rotation)
            angle_to_cell = math.atan2(dy, dx) + math.radians(rotation)
            angle_to_cell = (angle_to_cell + math.pi) % (2 * math.pi) - math.pi

            if -angle/2 <= angle_to_cell <= angle/2:
                pygame.draw.rect(screen, highlight_color, (col * cell_size, row * cell_size, cell_size, cell_size))
                highlighted_matrix[row, col] = 1

    return highlighted_matrix

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Increase rotation by 5 degrees when space is pressed
                player_rotation += 5
                player_rotation = (player_rotation + 180) % 360 - 180

    # Clear the screen
    screen.fill(background_color)

    # Draw the grid
    draw_grid()
    # Get the highlighted matrix
    highlighted_cells = draw_highlighted_cells(player_position, player_rotation)
    # Draw the player
    draw_player(player_position, player_rotation)
    #player_rotation += 1

    # Update the display
    pygame.display.flip()

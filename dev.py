import pygame
import sys
import numpy as np

# Initialize Pygame
pygame.init()

# Set the dimensions of the grid and rectangle
grid_size = 25
rectangle_size = 7

# Set the size of each cell in the grid
cell_size = 20

# Calculate the window size
width = height = grid_size * cell_size

# Set the color of the grid and highlighted cells
grid_color = (255, 255, 255)  # White
highlight_color = (255, 0, 0)  # Red

# Create the Pygame window
screen = pygame.display.set_mode((width, height))

def draw_rotated_rectangle(center, size, angle):
    # Calculate the rotated rectangle vertices
    rect_points = [
        np.array([-size // 2, -size // 2]),
        np.array([-size // 2, size // 2]),
        np.array([size // 2, size // 2]),
        np.array([size // 2, -size // 2]),
    ]

    # Rotate each point by the specified angle
    rotated_rect_points = [
        np.dot(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]), point) + center
        for point in rect_points
    ]

    # Draw the rotated rectangle
    pygame.draw.polygon(screen, highlight_color, rotated_rect_points, 0)

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the grid
    for row in range(grid_size):
        for col in range(grid_size):
            pygame.draw.rect(screen, grid_color, (col * cell_size, row * cell_size, cell_size, cell_size), 1)

    # Highlight cells to paint the 7x7 rectangle
    center = np.array([(9 + rectangle_size // 2) * cell_size, (9 + rectangle_size // 2) * cell_size])
    draw_rotated_rectangle(center, rectangle_size * cell_size, np.radians(45))

    # Update the display
    pygame.display.flip()

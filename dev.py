import pygame
import sys
import math

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
highlight_color = (255, 0, 0)  # Red

# Set the player's position (cell center)
player_position = (5.5, 25.5)

# Create the Pygame window
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Grid with Player")

def draw_grid():
    # Draw the grid
    for row in range(grid_size):
        for col in range(grid_size):
            pygame.draw.rect(screen, grid_color, (col * cell_size, row * cell_size, cell_size, cell_size), 1)

def draw_player(position):
    # Draw the player as a triangle
    x, y = position
    player_points = [(x * cell_size, y * cell_size),
                     ((x + 0.5) * cell_size, (y + 1) * cell_size),
                     (x * cell_size, (y + 1) * cell_size)]
    pygame.draw.polygon(screen, player_color, player_points)

def draw_highlighted_cells(position):
    x, y = position
    angle = math.radians(90)  # 90 degrees
    distance = 10  # Adjust this value for the desired range

    # Highlight cells within a 90-degree angle from the player's position
    for row in range(grid_size):
        for col in range(grid_size):
            dx = col - x
            dy = row - y
            angle_to_cell = math.atan2(dy, dx)
            distance_to_cell = math.sqrt(dx**2 + dy**2)

            if -angle/2 <= angle_to_cell <= angle/2 and distance_to_cell <= distance:
                pygame.draw.rect(screen, highlight_color, (col * cell_size, row * cell_size, cell_size, cell_size))

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Clear the screen
    screen.fill(background_color)

    # Draw the grid
    draw_grid()

    # Draw the player
    draw_player(player_position)

    # Highlight cells within a 90-degree angle from the player's position
    draw_highlighted_cells(player_position)

    # Update the display
    pygame.display.flip()
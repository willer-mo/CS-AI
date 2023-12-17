import pygame
import sys

# Initialize Pygame
pygame.init()

# Set the width and height of the screen
width, height = 640, 640
screen = pygame.display.set_mode((width, height))

# Set the color of the rectangle (R, G, B)
rect_color = (0, 255, 0)  # Green

# Set the dimensions of the rectangle
rect_width, rect_height = 100, 50

# Set the angle of rotation (45 degrees)
angle = 30

# Create a surface for the rectangle
rect_surface = pygame.Surface((rect_width, rect_height), pygame.SRCALPHA)
pygame.draw.rect(rect_surface, rect_color, (0, 0, rect_width, rect_height))

# Rotate the rectangle surface
rotated_rect_surface = pygame.transform.rotate(rect_surface, angle)

# Get the rectangle's position after rotation
rotated_rect_rect = rotated_rect_surface.get_rect(center=(width // 2, height // 2))

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the rotated rectangle
    screen.blit(rotated_rect_surface, rotated_rect_rect.topleft)

    # Update the display
    pygame.display.flip()

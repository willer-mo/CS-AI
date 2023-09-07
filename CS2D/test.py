import pygame
from pygame.math import Vector2

# Define the square vertices
square = [(30, 30), (700, 30), (700, 700), (30, 700)]

# Define points A and B
A = Vector2(330, 660)
B = Vector2(300, 630)

# Calculate the direction vector of the line
direction = B - A

# Initialize the intersection point
intersection = None

# Iterate over the square edges
for i in range(len(square)):
    # Define the edge points
    P1 = Vector2(square[i])
    P2 = Vector2(square[(i + 1) % len(square)])

    # Calculate the edge vector
    edge = P2 - P1

    # Check if the line and edge are parallel
    if direction.cross(edge) == 0:
        continue

    # Calculate the intersection point
    t = ((P1 - A).cross(edge)) / direction.cross(edge)
    point = A + t * direction

    # Check if the intersection point is on the edge
    if min(P1.x, P2.x) <= point.x <= max(P1.x, P2.x) and min(P1.y, P2.y) <= point.y <= max(P1.y, P2.y):
        intersection = point
        break

# Print the intersection point
if intersection:
    print(f"The intersection point is: {intersection}")
else:
    print("No intersection found")

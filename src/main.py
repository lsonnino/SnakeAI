################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
# The project:
#     The goal of this project is to make an AI that learns to play snake
#
# Requirements:
#     * PyGame: to display and play snake
#     * TensorFlow: manages the AI
#     * Numpy: used to manage matrix
#
# This file is the main file -- this is the file that runs the game
# (the constants file indicates whether the AI plays or not)
#
################################################################

from src.objects import *
from src.constants import *


# Start the game
pygame.init()
# Initialize the fonts
pygame.font.init()

# Creating the window
window = pygame.display.set_mode(WIN_SIZE)

# Setting the window name
pygame.display.set_caption(NAME)

# Setting up the clock
clock = pygame.time.Clock()


# Get the font
font = pygame.font.SysFont(FONT, FONT_SIZE)


# Creates the map
map = Map()
map.spawn_food()


# AI variables
ai_generation = 1

# Keeps the game running
running = True
while running:
    # pygame.time.delay(FRAME_TIME)

    for event in pygame.event.get():
        # Check special events
        if event.type == pygame.QUIT: # Quit
            running = False
            break

        # Check pressed keys
        keys = pygame.key.get_pressed()

        for key in keys:
            if keys[pygame.K_ESCAPE]:
                running = False
                break

            if AI:
                continue

            elif keys[pygame.K_LEFT]:
                map.snake.direction = LEFT
            elif keys[pygame.K_RIGHT]:
                map.snake.direction = RIGHT
            elif keys[pygame.K_UP]:
                map.snake.direction = TOP
            elif keys[pygame.K_DOWN]:
                map.snake.direction = BOTTOM

    # Make the snake move
    if map.snake.walk():
        # The snake is still alive
        pass
    else:
        # The snake died
        running = False

    map.check_food()

    # Draw the components
    map.draw(window)

    # Draw the texts
    textsurface = font.render("Score: " + str(map.snake.get_score()), False, TEXT_COLOR)
    # Merge the texts with the window
    window.blit(textsurface, (10, 10))
    if AI:
        textsurface = font.render("Generation: " + str(ai_generation), False, TEXT_COLOR)
        # Merge the texts with the window
        window.blit(textsurface, (10, WIN_SIZE[1] - 10 - FONT_SIZE))

    # Refresh the window
    pygame.display.flip()

    # Wait until next frame
    clock.tick(FPS)

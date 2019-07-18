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
from src.ai import *

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

# setup AI
if AI_PLAYS:
    ai_generation = 1
    network = DeepQNetwork()
    ai = AI(EpsilonGreedyStrategy(), 5)

map = Map()

# Keeps the game running
running = True
while running:
    # Creates the map
    map.__init__()
    map.spawn_food()

    # Starts playing
    playing = True

    # initialize reward
    reward = 0

    while playing:
        for event in pygame.event.get():
            # Check special events
            if event.type == pygame.QUIT:  # Quit
                running = False
                playing = False
                break

        # Check pressed keys
        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            running = False
            playing = False
            break

        # Get action
        action = NONE
        if AI_PLAYS:
            # The ai returns a number between 0 and 5 (from NONE to BOTTOM as described in the constants)
            action = ai.select_action(
                torch.from_numpy(map_to_input(map)).float(),
                network
            )
        else:
            if keys[pygame.K_LEFT]:
                action = LEFT
            elif keys[pygame.K_RIGHT]:
                action = RIGHT
            elif keys[pygame.K_UP]:
                action = TOP
            elif keys[pygame.K_DOWN]:
                action = BOTTOM

        # Perform action
        if action != NONE:
            map.snake.direction = action

        # Make the snake's move
        if not map.snake.walk():
            # The snake died
            playing = False
            reward = -1
            continue

        if map.check_food():
            reward = 1
        else:
            reward = 0

        # Draw the components
        map.draw(window)

        # Draw the texts
        textsurface = font.render("Score: " + str(map.snake.get_score()), False, TEXT_COLOR)
        # Merge the texts with the window
        window.blit(textsurface, (10, 10))
        if AI_PLAYS:
            textsurface = font.render("Generation: " + str(ai_generation), False, TEXT_COLOR)
            # Merge the texts with the window
            window.blit(textsurface, (10, WIN_SIZE[1] - 10 - FONT_SIZE))

        # Refresh the window
        pygame.display.flip()

        # Wait until next frame
        clock.tick(FPS)

    # Pass to next generation
    if AI_PLAYS:
        ai_generation += 1

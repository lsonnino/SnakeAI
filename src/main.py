################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
# The project:
#     The goal of this project is to make an AI that learns to play snake
#     using reinforcement learning -- by giving him a reward whenever he
#     does something good and by punishing him when he does something bad.
#
# Requirements:
#     * PyGame     : to display and play snake
#     * TensorFlow : manages the AI
#     * Numpy      : used to manage matrix
#
# This file is the main file -- this is the file that runs the game
# (the constants file indicates whether the AI plays or not)
#
################################################################

from src.game import *

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
    player = AIPlayer(DEVICE)
else:
    player = HumanPlayer()

# Create the game
game = Game(player, DEVICE, max_moves=(AI_MAX_ALLOWED_MOVES if AI_PLAYS else -1))

# Keeps the game running
running = True
gameNum = 0  # keeps track of the number of played games
while running and (NUMBER_OF_GAMES < 0 or gameNum < NUMBER_OF_GAMES):
    # Reset the game
    game.reset()

    while game.playing:
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

        # Make a step in the game
        reward = game.step()

        # Draw the components
        game.draw(window)

        # Draw the texts
        text_surface = font.render("Score: " + str(game.get_score()), False, TEXT_COLOR)
        # Merge the texts with the window
        window.blit(text_surface, (10, 10))
        if AI_PLAYS:
            text_surface = font.render("Generation: " + str(ai_generation), False, TEXT_COLOR)
            # Merge the texts with the window
            window.blit(text_surface, (10, WIN_SIZE[1] - 10 - FONT_SIZE))

        # Refresh the window
        pygame.display.flip()

        # Wait until next frame
        clock.tick(FPS)

    # Pass to next generation
    if AI_PLAYS:
        # Printing score
        print("AI score for gen " + str(ai_generation) + ": " + str(game.get_score()))

        ai_generation += 1
    else:
        # Printing score
        print("Game score: " + str(game.get_score()))

    gameNum += 1

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


def ai_action_to_text(direction):
    if direction == RIGHT:
        return "Right"
    elif direction == LEFT:
        return "Left"
    elif direction == BOTTOM:
        return "Down"
    elif direction == TOP:
        return "Up"
    else:
        return "None"

def main():
    if GRAPHICS:
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
        player = AIPlayer(AI_MODEL_BUILDER)
    else:
        player = HumanPlayer()

    game_num = 1  # keeps track of the number of played games

    # Create the game
    initial_food_spawn = max(1, INITIAL_FOOD_SPAWN)
    game = Game(player, STATE_BUILDER, EMPTY_STATE_BUILDER,
                max_moves=(AI_MAX_ALLOWED_MOVES if AI_PLAYS else -1), initial_food_spawn=initial_food_spawn)
    # Load the snake
    if AI_PLAYS and LOAD_NUMBER >= 0:
        if os.path.exists(get_path(LOAD_NUMBER)):
            read_ai_num(game.player, LOAD_NUMBER)
            game_num = LOAD_NUMBER + 1
            ai_generation = LOAD_NUMBER

    if AI_PLAYS:
        game.player.brain.model.summary()

    # Keeps the game running
    running = True
    while running and (NUMBER_OF_GAMES < 0 or game_num < NUMBER_OF_GAMES):
        # Reset the game
        game.reset()
        state = game.get_state()

        last_score = 0
        while game.playing:
            if GRAPHICS:
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
            action, reward = game.step()

            # Train the player
            next_state = game.get_state()
            game.set_result(state, action, reward, next_state)
            game.train()
            state = next_state

            last_score = max(last_score, game.get_score())

            if GRAPHICS:
                # Draw the components
                game.draw(window)

                # Draw the texts
                text_surface = font.render("Score: " + str(last_score), False, TEXT_COLOR)
                # Merge the texts with the window
                window.blit(text_surface, (10, 10))
                if AI_PLAYS:
                    # Generation
                    text_surface = font.render("Generation: " + str(ai_generation), False, TEXT_COLOR)
                    # Merge the texts with the window
                    window.blit(text_surface, (10, WIN_SIZE[1] - 10 - FONT_SIZE))

                    # Action
                    text = "Action: " + ai_action_to_text(game.map.snake.direction)
                    text_surface = font.render(text, False, TEXT_COLOR)
                    # Merge the texts with the window
                    window.blit(text_surface, (10, WIN_SIZE[1] - 20 - 2 * FONT_SIZE))

                # Refresh the window
                pygame.display.flip()

                # Wait until next frame
                clock.tick(FPS)

        # Pass to next generation
        if AI_PLAYS:
            # Printing score
            print("AI score for gen " + str(ai_generation) + ": " + str(last_score))
            step = game.player.iteration
            print("current step: " + str(step) + " - greed: " + str(game.player.epsilon))

            ai_generation += 1
        else:
            # Printing score
            print("Game score: " + str(last_score))

        # Save the ai
        if AI_PLAYS and game_num % SAVE_EVERY == 0:
            save_ai_num(game.player, game_num)

        game_num += 1

        game.next_episode()

    if AI_PLAYS:
        save_ai_num(game.player, game_num)


snake_ai = compile('main()', 'snake_ai', 'exec')

exec(snake_ai)

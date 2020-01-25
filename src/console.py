################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
################################################################

from src.game import *


game = None
clock = None
font = None
window = None
ai_generation = -1


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


def boot():
    global window, game, clock, font, ai_generation

    if GRAPHICS:
        # Start the game
        pygame.init()
        # Initialize the fonts
        pygame.font.init()

        # Creating the window
        window = pygame.display.set_mode(WIN_SIZE)

        # Setting the window name
        pygame.display.set_caption(NAME + " - " + AI_NAME)

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

    # Create the game
    initial_food_spawn = max(1, INITIAL_FOOD_SPAWN)
    game = Game(player, STATE_BUILDER, EMPTY_STATE_BUILDER,
                max_moves=(AI_MAX_ALLOWED_MOVES if AI_PLAYS else -1), initial_food_spawn=initial_food_spawn)
    # Load the snake
    if AI_PLAYS and LOAD_NUMBER >= 0:
        if read_ai_num(game.player, LOAD_NUMBER):
            ai_generation = LOAD_NUMBER + 1


def play():
    last_score = 0
    stopped = False

    state = game.get_state()

    while game.playing:
        if GRAPHICS:
            for event in pygame.event.get():
                # Check special events
                if event.type == pygame.QUIT:  # Quit
                    game.playing = False
                    stopped = True
                    break

            # Check pressed keys
            keys = pygame.key.get_pressed()

            if keys[pygame.K_ESCAPE]:
                game.playing = False
                stopped = True
                break
            elif keys[pygame.K_SPACE]:  # Pause the game
                continue

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

    return last_score, stopped

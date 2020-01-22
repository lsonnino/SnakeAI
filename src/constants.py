################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
# This file contains every needed configuration and
# global variables
#
################################################################

# General Constants
NAME = "SnakeAI"
AI_PLAYS = True
NUMBER_OF_GAMES = -1  # -1 to play infinite games
SAVE_EVERY = 500  # Must be greater than 0
LOAD_NUMBER = 1000  # -1 to start from the beginning
DATA_DIR = "data"
EXTENSION = "snake"
PARAMS_EXTENSION = "params"
GRAPHICS = False
RESET_GREED = False

FONT = "Arial"
FONT_SIZE = 20

# World constants
WIN_SIZE = (760, 760)  # The size of the window
ROWS = 20  # The number of rows (38 is a good size)
COLUMNS = 20  # The number of columns (38 is a good size)
CASE_WIDTH = WIN_SIZE[0] / COLUMNS
CASE_HEIGHT = WIN_SIZE[1] / ROWS

INITIAL_FOOD_SPAWN = 1  # The number of apples that will spawn at game launch

FPS_AI = 120  # The used frame rate if the AI plays (the bigger, the faster it plays)
FPS_HUMAN = 10  # The used frame rate if the human plays
if AI_PLAYS:
    FPS = FPS_AI
else:
    FPS = FPS_HUMAN

# Game style -- every color is defined in here
EMPTY_COLOR = (230, 230, 240)
FOOD_COLOR = (255, 59, 48)
SNAKE_COLOR = (0, 122, 255)
TEXT_COLOR = (77, 76, 76)
colors = [EMPTY_COLOR, FOOD_COLOR]

# Objects constants
EMPTY = 0
FOOD = 1
SNAKE = 2

RIGHT = 0
TOP = 1
LEFT = 2
BOTTOM = 3
NONE = 4

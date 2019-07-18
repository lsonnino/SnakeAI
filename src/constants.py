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
AI_PLAYS = False

FONT = "Arial"
FONT_SIZE = 20

# World constants
WIN_SIZE = (750, 750)  # The size of the window
ROWS = 30  # The number of rows
COLUMNS = 30  # The number of columns
CASE_WIDTH = WIN_SIZE[0] / COLUMNS
CASE_HEIGHT = WIN_SIZE[1] / ROWS

FPS_AI = 30  # The used frame rate if the AI plays (the bigger, the faster it plays)
FPS_HUMAN = 10  # The used frame rate if the human plays
if AI_PLAYS:
    FPS = FPS_AI
else:
    FPS = FPS_HUMAN

# Game style -- each color is defined in here
EMPTY_COLOR = (230, 230, 240)
FOOD_COLOR = (255, 59, 48)
SNAKE_COLOR = (0,122,255)
TEXT_COLOR = (77, 76, 76)
colors = [EMPTY_COLOR, FOOD_COLOR]

# Objects constants
EMPTY = 0
FOOD = 1
SNAKE = 2

RIGHT = 1
TOP = 2
LEFT = 3
BOTTOM = 4
NONE = 0

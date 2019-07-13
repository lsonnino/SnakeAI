# General Constants
NAME = "SnakeAI"
AI = True

FONT = "Arial"
FONT_SIZE = 20

# World constants
WIN_SIZE = (500, 500)
ROWS = 20
COLUMNS = 20
CASE_WIDTH = WIN_SIZE[0] / ROWS
CASE_HEIGHT = WIN_SIZE[1] / COLUMNS

FPS_AI = 30
FPS_HUMAN = 10
if AI:
    FPS = FPS_AI
else:
    FPS = FPS_HUMAN

# Game style
EMPTY_COLOR = (230, 230, 240)
FOOD_COLOR = (255, 59, 48)
SNAKE_COLOR = (0,122,255)
colors = [EMPTY_COLOR, FOOD_COLOR]
TEXT_COLOR = (77, 76, 76)

# Objects constants
EMPTY = 0
FOOD = 1
SNAKE = 2

RIGHT = 0
TOP = 1
LEFT = 2
BOTTOM = 3
NONE = -1

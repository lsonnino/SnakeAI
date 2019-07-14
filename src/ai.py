################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
# This file contains the AI
#
################################################################

import numpy as np
from src.constants import *


def map_to_input(map):
    """
    Get the input value as they will be given to the AI
    The input is an array of size COLUMNS * ROWS + 2 where the first COLUMNS * ROWS values
    represents the map. For each input, the value is 1 if the snake is at that position, 0.5 if there is food and
    0 otherwise.
    The last two inputs are the snake's head position. The x and y position are a number between 0 and 1. 0 if the
    head is at the left/up most position on the map, 1 if the head is at the right/bottom most position.

    :param map: the current state of the map the AI is playing
    :return: the input value as they will be given to the AI
    """
    inputs = np.zeros(COLUMNS * ROWS + 2)

    # Runs the map
    for x in range(COLUMNS):
        for y in range(ROWS):
            if map.map[x, y] == FOOD:
                value = 0.5
            else:
                value = 1

            inputs[x * COLUMNS + y] = value

    # Add the snake
    for piece in map.snake.body:
        inputs[piece[0] * COLUMNS + piece[1]] = 1

    # Add the head's position
    head_x, head_y = get_head_position(map.snake)
    inputs[COLUMNS * ROWS] = head_x
    inputs[COLUMNS * ROWS + 1] = head_y

    return inputs


def get_head_position(snake):
    """
    The last two inputs are the snake's head position. The x and y position are a number between 0 and 1. 0 if the
    head is at the left/up most position on the map, 1 if the head is at the right/bottom most position.

    :param map: the snake the AI is playing
    :return: a tuple containing first the x position then the y position
    """
    x = snake.body[0][0] / COLUMNS
    y = snake.body[0][1] / ROWS

    return x, y

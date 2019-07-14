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
    The input is an array of size COLUMNS * ROWS + 3 where the first COLUMNS * ROWS values
    represents the map. For each input, the value is 2 if the snake is at that position, 1 if there is food and
    0 otherwise.
    The last three inputs are the snake's head position and the direction the snake is looking at.

    :param map: the current state of the map the AI is playing
    :return: the input value as they will be given to the AI
    """
    inputs = np.zeros(COLUMNS * ROWS + 3)

    # Runs the map
    for x in range(COLUMNS):
        for y in range(ROWS):
            if map.map[x, y] == FOOD:
                value = 1
            else:
                value = 0

            inputs[x * COLUMNS + y] = value

    # Add the snake
    for piece in map.snake.body:
        inputs[piece[0] * COLUMNS + piece[1]] = 2

    # Add the head's position
    head_x, head_y = get_head_position(map.snake)
    inputs[COLUMNS * ROWS] = head_x
    inputs[COLUMNS * ROWS + 1] = head_y

    inputs[COLUMNS * ROWS + 2] = map.snake.direction

    return inputs


def get_head_position(snake):
    """
    Get the snake's head position

    :param map: the snake the AI is playing
    :return: a tuple containing first the x position then the y position
    """
    x = snake.body[0][0]
    y = snake.body[0][1]

    return x, y


def get_output(outputs):
    """
    Interprets the result of the neural network to work out the next step.
    If the neural network is not sure, it does nothing

    :param outputs: the outputs of the neural network. An array of five values:
        The first one is its will to do nothing
        The second one is its will to go right
        The third one is its will to go up
        The fourth one is its will to go left
        The fifth one is its will to go down
    :return: The direction the snake will go (RIGHT, TOP, LEFT, BOTTOM or NONE)
    """
    index = 0
    multiple = False
    value = 0

    for i in range(len(outputs)):
        if outputs[i] > value:
            value = outputs[i]
            index = i
            multiple = False
        elif outputs[i] == value:
            multiple = True

    if multiple:
        return NONE

    if index == 1:  # RIGHT
        return RIGHT
    elif index == 2:  # TOP
        return TOP
    elif index == 3:  # LEFT
        return LEFT
    elif index == 4:  # BOTTOM
        return BOTTOM
    else:  # None
        return NONE


class AI(object):
    def __init__(self):
        pass

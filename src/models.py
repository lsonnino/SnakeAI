################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
################################################################

from src.constants import *

from tensorflow import keras
import numpy as np


#
# AI MODEL BUILDERS
#


def omniscient_ai_model_builder():
    number_of_actions = 4

    inputs = keras.Input(shape=(COLUMNS * ROWS + 4,), name='input')
    # x = keras.layers.Dense(COLUMNS * ROWS / 2, activation='linear', name='hidden_layer')(inputs)
    # outputs = keras.layers.Dense(self.number_of_actions, activation='relu', name='output')(x)
    outputs = keras.layers.Dense(number_of_actions, activation='relu', name='output')(inputs)

    return number_of_actions, inputs, outputs, "SnakeAI"


def four_directional_ai_model_builder():
    number_of_actions = 4

    inputs = keras.Input(shape=(3,), name='input')
    outputs = keras.layers.Dense(number_of_actions, activation='relu', name='output')(inputs)

    return number_of_actions, inputs, outputs, "SnakeAI"



#
# AI MODEL BUILDERS
#


def entire_map_to_state(map):
    state = np.zeros(COLUMNS * ROWS)

    # Runs the map
    for x in range(COLUMNS):
        for y in range(ROWS):
            if map.map[x, y] == FOOD:
                value = 1
            else:
                value = 0

            state[x * ROWS + y] = value

    # Add the snake
    for piece in map.snake.body:
        state[piece[0] * ROWS + piece[1]] = -1

    head = map.snake.body[0]
    state[head[0] * ROWS + head[1]] = -2

    return state


def omniscient_empty_state_builder():
    return np.zeros(COLUMNS * ROWS)


def merge_entire_states(new, direction):
    """
    merged = np.zeros( (COLUMNS, ROWS, 2) )

    for x in range(COLUMNS):
        for y in range(ROWS):
            merged[x, y] = [new[x, y], previous[x, y]]
    """
    direction_array = np.zeros(4)
    if direction != NONE:
        direction_array[direction] = 1

    merged = np.concatenate((new, direction_array))

    return merged


def omniscient_state_builder(map, playing_or_starting):
    if playing_or_starting:
        return merge_entire_states(omniscient_empty_state_builder(), NONE)
    else:
        return merge_entire_states(entire_map_to_state(map), map.snake.direction)

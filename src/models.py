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
# UTILS
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


def snake_direction_to_array(direction):
    direction_array = np.zeros(4)
    if direction != NONE:
        direction_array[direction] = 1

    return direction_array


def merge_states(first, second):
    """
    merged = np.zeros( (COLUMNS, ROWS, 2) )

    for x in range(COLUMNS):
        for y in range(ROWS):
            merged[x, y] = [new[x, y], previous[x, y]]
    """

    merged = np.concatenate((first, second))

    return merged


def get_distance_from_obstacle(map, start, dir):
    dx, dy = dir
    x, y = start
    x += dx
    y += dy
    count = 1

    while (0 <= x < COLUMNS and 0 <= y < ROWS) and (map.map[x, y] == EMPTY or map.map[x, y] == FOOD):
        x += dx
        y += dy
        count += 1

    return count


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


def tri_directional_ai_model_builder():
    number_of_actions = 4

    inputs = keras.Input(shape=(10,), name='input')
    outputs = keras.layers.Dense(number_of_actions, activation='relu', name='output')(inputs)

    return number_of_actions, inputs, outputs, "SnakeAI"


#
# STATE MODEL BUILDERS
#

def omniscient_empty_state_builder():
    return np.zeros(COLUMNS * ROWS)


def omniscient_state_builder(map, alive, first):
    if alive or first:
        return merge_states(omniscient_empty_state_builder(), snake_direction_to_array(NONE))
    else:
        return merge_states(entire_map_to_state(map), snake_direction_to_array(map.snake.direction))


def tri_directional_empty_state_builder():
    return np.zeros(6)


def tri_directional_state_builder(map, alive, first):
    if first:
        return merge_states(tri_directional_empty_state_builder(), snake_direction_to_array(NONE))
    elif not alive:
        return merge_states(tri_directional_empty_state_builder(), snake_direction_to_array(map.snake.direction))

    looking_direction = map.snake.direction
    head = map.snake.body[0]

    if looking_direction == RIGHT:
        direction_1 = (0, -1)
        direction_2 = (1, 0)
        direction_3 = (0, 1)
    elif looking_direction == BOTTOM:
        direction_1 = (1, 0)
        direction_2 = (0, 1)
        direction_3 = (-1, 0)
    elif looking_direction == LEFT:
        direction_1 = (0, 1)
        direction_2 = (-1, 0)
        direction_3 = (0, -1)
    else:  # TOP or NONE
        direction_1 = (-1, 0)
        direction_2 = (0, -1)
        direction_3 = (1, 0)

    distances = [
        get_distance_from_obstacle(map, head, direction_1),
        get_distance_from_obstacle(map, head, direction_2),
        get_distance_from_obstacle(map, head, direction_3)
    ]

    food_distance = [
        head[0] - map.food_position[0],
        head[1] - map.food_position[1],
    ]

    tmp_state = merge_states(distances, food_distance)
    tmp_state = merge_states(tmp_state, [map.snake.get_score()])
    return merge_states(tmp_state, snake_direction_to_array(looking_direction))

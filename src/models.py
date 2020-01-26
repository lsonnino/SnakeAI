################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
################################################################

from src.constants import *

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


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
    count = 0

    while (0 <= x < COLUMNS and 0 <= y < ROWS) and (map.map[x, y] == EMPTY or map.map[x, y] == FOOD) and \
            (x, y) not in map.snake.body:
        x += dx
        y += dy
        count += 1

    # count += 1  # If removed, could it improve the skills?

    return count


#
# AI MODEL BUILDERS
#


def omniscient_ai_model_builder():
    number_of_actions = 4
    input_dimension = [COLUMNS * ROWS + 4]

    input = tf.placeholder(tf.float32, shape=[None, *input_dimension], name='inputs')
    actions = tf.placeholder(tf.float32, shape=[None, number_of_actions], name='actions_taken')
    q_target = tf.placeholder(tf.float32, shape=[None, number_of_actions], name='q_values')

    flat = tf.layers.flatten(input)
    Q_values = tf.layers.dense(flat, units=number_of_actions)

    return input, actions, q_target, Q_values


def base_ai_model_builder(number_of_actions, input_dimension):
    input = tf.placeholder(tf.float32, shape=[None, *input_dimension], name='inputs')
    actions = tf.placeholder(tf.float32, shape=[None, number_of_actions], name='actions_taken')
    q_target = tf.placeholder(tf.float32, shape=[None, number_of_actions], name='q_values')

    flat = tf.layers.flatten(input)
    dense1 = tf.layers.dense(flat, units=32, activation=tf.nn.relu)
    dense2 = tf.layers.dense(dense1, units=32, activation=tf.nn.relu)
    Q_values = tf.layers.dense(dense2, units=number_of_actions)

    return input, actions, q_target, Q_values


def tri_directional_ai_model_builder():
    return base_ai_model_builder(number_of_actions=4, input_dimension=[10])


def four_directional_ai_model_builder():
    return base_ai_model_builder(number_of_actions=4, input_dimension=[7])


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
    if not alive:
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


def four_directional_empty_state_builder():
    return np.zeros(7)


def four_directional_state_builder(map, alive, first):
    if not alive:
        return four_directional_empty_state_builder()

    head = map.snake.body[0]

    distances = [
        get_distance_from_obstacle(map, head, ( 1,  0)),
        get_distance_from_obstacle(map, head, (-1,  0)),
        get_distance_from_obstacle(map, head, ( 0,  1)),
        get_distance_from_obstacle(map, head, ( 0, -1))
    ]

    food_distance = [
        head[0] - map.food_position[0],
        head[1] - map.food_position[1],
    ]

    tmp_state = merge_states(distances, food_distance)
    return merge_states(tmp_state, [map.snake.get_score()])

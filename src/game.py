################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
# This file contains the game himself
#
################################################################

import os
import pickle

from src.objects import *
from src.ai import *


def map_to_state(map):
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
        state[piece[0]* ROWS + piece[1]] = -1

    return state


def get_empty_state():
    return np.zeros(COLUMNS * ROWS)


def merge_states(new, previous):
    '''
    merged = np.zeros( (COLUMNS, ROWS, 2) )

    for x in range(COLUMNS):
        for y in range(ROWS):
            merged[x, y] = [new[x, y], previous[x, y]]
    '''

    merged = np.concatenate((previous, new))

    return merged


class Game(object):
    def __init__(self, player, max_moves=-1, initial_food_spawn=1):
        """
        Constructor of the game
        :param player: the player
        :param max_moves: the maximum number of moves the snake is allowed to perform to get some food
                before dying. -1 means that there is no maximum.
        """
        self.max_moves = max_moves
        self.player = player
        self.map = Map(max_moves=max_moves)
        self.map.spawn_food()
        self.prev_state = get_empty_state()

        self.initial_food_spawn = initial_food_spawn

        self.playing = True
        self.starting = True

    def reset(self):
        """
        Reset the game to its initial state
        """
        self.map.__init__(max_moves=self.max_moves)
        for n in range(self.initial_food_spawn):
            self.map.spawn_food()

        self.prev_state = get_empty_state()

        self.playing = True
        self.starting = True

    def step(self):
        """
        Make a step in the game. Get the action to perform from the player and check the result
        :return: the action and current reward
        """
        reward_val = 0

        # Get the player's action
        action = self.player.get_action(self.get_state())
        if action != NONE:
            self.map.snake.direction = action

        self.prev_state = map_to_state(self.map)

        # Make the snake's move and check result
        if not self.map.snake.walk():
            # The snake died
            self.playing = False
            reward_val = -1
        elif self.map.check_food():  # The snake got some food
            self.map.snake.got_food()
            reward_val = 1

        if self.starting:
            self.starting = False

        # Return the player action and reward
        return action, reward_val

    def get_state(self):
        if self.playing or self.starting:
            return merge_states(map_to_state(self.map), self.prev_state)
        else:
            return get_empty_state()

    def train(self):
        self.player.train()

    def set_result(self, state, action, reward, next_state):
        self.player.set_result(state, action, reward, next_state, self.playing)

    def draw(self, window):
        """
        Draws the game in the window
        :param window: the surface to draw on
        """
        window.fill(EMPTY_COLOR)

        self.map.draw(window)

    def get_score(self):
        """
        Get the current score
        :return: the player's score
        """
        return self.map.snake.get_score()

    def next_episode(self):
        self.player.reset()


# ============================
# = Players
# ============================

class HumanPlayer:
    def __init__(self):
        pass

    def get_action(self, state):
        keys = pygame.key.get_pressed()

        action = NONE
        if keys[pygame.K_LEFT]:
            action = LEFT
        elif keys[pygame.K_RIGHT]:
            action = RIGHT
        elif keys[pygame.K_UP]:
            action = TOP
        elif keys[pygame.K_DOWN]:
            action = BOTTOM

        return action

    def train(self):
        pass

    def set_result(self, state, action, reward, next_state, done):
        pass

    def reset(self):
        pass


class AIPlayer:
    def __init__(self):
        self.brain = AI()
        self.target = AI()
        self.iteration = 0
        self.epsilon = max_exploration_rate

    def get_action(self, state):
        self.iteration += 1

        return self.brain.get_action(
            state,
            self.epsilon
        )

    def train(self):
        self.brain.train(self.target)

        if self.iteration % update_frequency:
            self.target.copy_weights(self.brain)

    def set_result(self, state, action, reward, next_state, done):
        self.brain.add_experience(state, action, reward, next_state, done)

    def reset(self):
        self.iteration = 0
        self.target.copy_weights(self.brain)
        self.epsilon = max(min_exploration_rate, self.epsilon * exploration_decay_rate)


class ParamsSerializer(object):
    def __init__(self, epsilon, experience):
        self.epsilon = epsilon
        self.experience = experience


def get_path(num):
    return DATA_DIR + '/' + str(num) + '.' + EXTENSION


def get_params_path(num):
    return DATA_DIR + '/' + str(num) + '.' + PARAMS_EXTENSION


def read_ai_num(player, num):
    player.brain.model = tf.keras.models.load_model(get_path(num))

    with open(get_params_path(num), 'rb') as f:
        params_serializer = pickle.load(f)
        player.epsilon = params_serializer.epsilon


def save_ai_num(ai, num):
    path = get_path(num)

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    elif os.path.exists(path):
        os.remove(path)

    ai.brain.model.save(path)

    path = get_params_path(num)
    if os.path.exists(path):
        os.remove(path)

    with open(path, 'wb') as f:
        params_serializer = ParamsSerializer(ai.epsilon, ai.brain.experience)
        pickle.dump(params_serializer, f)

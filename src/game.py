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


class Game(object):
    def __init__(self, player, state_builder, empty_state_builder, max_moves=-1, initial_food_spawn=1):
        """
        Constructor of the game
        :param player: the player
        :param state_builder: a function that creates a state from the map to feed the AI
        :param empty_state_builder: a function that creates an empty state to feed the AI
        :param max_moves: the maximum number of moves the snake is allowed to perform to get some food
                before dying. -1 means that there is no maximum.
        :param initial_food_spawn: the number of food pieces that will spawn
        """
        self.empty_state_builder = empty_state_builder
        self.state_builder = state_builder

        self.max_moves = max_moves
        self.player = player
        self.map = Map(max_moves=max_moves)
        self.map.spawn_food()
        self.prev_state = self.empty_state_builder()

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
        return self.state_builder(self.map, self.playing or self.starting)

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
    def __init__(self, ai_model_builder):
        self.brain = AI(ai_model_builder)
        self.target = AI(ai_model_builder)
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

        if self.epsilon > ss_thresh:
            self.epsilon = max(ss_thresh, self.epsilon * ss_exploration_decay_rate)
        else:
            self.epsilon = max(min_exploration_rate, self.epsilon * exploration_decay_rate)


class ParamsSerializer(object):
    def __init__(self, epsilon, experience, model):
        self.epsilon = epsilon
        self.experience = experience
        self.model = model


def get_path(num):
    return DATA_DIR + '/' + str(num) + '.' + EXTENSION


def get_params_path(num):
    return DATA_DIR + '/' + str(num) + '.' + PARAMS_EXTENSION


def read_ai_num(player, num):
    with open(get_path(num), 'rb') as f:
        params_serializer = pickle.load(f)
        player.brain.model = tf.keras.models.model_from_json(params_serializer.model)
        if not RESET_GREED:
            player.epsilon = params_serializer.epsilon


def save_ai_num(ai, num):
    path = get_path(num)

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    elif os.path.exists(path):
        os.remove(path)

    json_string = ai.brain.model.to_json()

    with open(path, 'wb') as f:
        params_serializer = ParamsSerializer(ai.epsilon, ai.brain.experience, json_string)
        pickle.dump(params_serializer, f)

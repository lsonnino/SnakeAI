################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
# This file contains the game himself
#
################################################################

import os

from src.objects import *
from src.ai import *
from math import sqrt, pow


def get_distance_from_food(map):
    head = map.snake.body[0]
    tmp = (head[0] - map.food_position[0])**2 + (head[1] - map.food_position[1])**2
    return sqrt(tmp)


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
        old_distance_from_food = get_distance_from_food(self.map)

        # Get the player's action
        action = self.player.get_action(self.get_state())
        if action != NONE:
            self.map.snake.direction = action

        # Make the snake's move and check result
        if not self.map.snake.walk():
            # The snake died
            self.playing = False
            reward_val -= 100
        elif self.map.check_food():  # The snake got some food
            self.map.snake.got_food()
            reward_val += 50 * self.map.snake.get_score()
        else:
            weight = 25 * pow(0.9, self.map.snake.get_score())
            if weight < 1:
                weight = 0
            distance_from_food = get_distance_from_food(self.map)
            distance_delta = distance_from_food - old_distance_from_food
            reward_val += weight if distance_delta < 0 else -weight

        if len(self.map.snake.body) == COLUMNS * ROWS:  # The game ended
            reward_val = 10000

        if self.starting:
            self.starting = False

        # Return the player action and reward
        return action, reward_val

    def get_state(self):
        return self.state_builder(self.map, self.playing, self.starting)

    def train(self):
        self.player.train()

    def set_result(self, state, action, reward, next_state):
        self.player.set_result(state, action, reward, next_state, not self.playing)

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
        self.brain = Agent(n_actions=4, name=AI_NAME, input_dims=INPUT_DIMENSION, network_builder=ai_model_builder)
        self.iteration = 0

    def get_action(self, state):
        self.iteration += 1

        return self.brain.choose_action(state)

    def train(self):
        self.brain.learn()

    def set_result(self, state, action, reward, next_state, done):
        self.brain.store_transition(state=state, chosen_action=action, reward=reward, new_state=next_state, terminal=done)

    def reset(self):
        self.iteration = 0


def get_path(num):
    return DATA_DIR + '/' + str(num) + '.' + EXTENSION


def read_ai_num(player, num):
    path = get_path(num)
    try:
        player.brain.load_models(path)
        return True
    except Exception:
        return False


def save_ai_num(player, num):
    path = get_path(num)

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    elif os.path.exists(path):
        os.remove(path)

    player.brain.save_models(path)


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

from torch import optim

from src.objects import *
from src.ai import *


def map_to_state(map):
    state = np.zeros( (COLUMNS, ROWS) )

    # Runs the map
    for x in range(COLUMNS):
        for y in range(ROWS):
            if map.map[x, y] == FOOD:
                value = 1
            else:
                value = 0

            state[x, y] = value

    # Add the snake
    for piece in map.snake.body:
        state[piece[0], piece[1]] = -1

    return state


def get_empty_state():
    return np.zeros( (COLUMNS, ROWS) )


def merge_states(new, previous):
    merged = np.zeros( (COLUMNS, ROWS, 2) )

    for x in range(COLUMNS):
        for y in range(ROWS):
            merged[x, y] = [new[x, y], previous[x, y]]

    return merged


class Game(object):
    def __init__(self, player, max_moves=-1):
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

        self.playing = True
        self.starting = True

    def reset(self):
        """
        Reset the game to its initial state
        """
        self.map.__init__(max_moves=self.max_moves)
        self.map.spawn_food()

        self.prev_state = get_empty_state()

        self.playing = True
        self.starting = True

    def step(self):
        """
        Make a step in the game. Get the action to perform from the player and check the result
        :return: the reward the player got from its action
        """
        reward_val = 0

        # Get the player's action
        action = self.player.get_action(self.get_state())
        if action.item() != NONE:
            self.map.snake.direction = action.item()

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

        # Return the reward
        return torch.tensor([action], device=DEVICE), torch.tensor([reward_val], device=DEVICE)
        # return action, reward_val

    def get_state(self):
        if self.playing or self.starting:
            return merge_states(map_to_state(self.map), self.prev_state)
        else:
            return np.zeros( (COLUMNS, ROWS, 2) )

    def train(self):
        self.player.train()

    def set_result(self, state, action, reward, next_state):
        self.player.set_result(state, action, reward, next_state)

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

    def next_episode(self, number):
        self.player.next_episode(number)


# ============================
# = Players
# ============================

class HumanPlayer:
    def __init__(self):
        pass

    def get_action(self, map):
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

    def set_result(self, state, action, reward, next_state):
        pass

    def next_episode(self, number):
        pass


class AIPlayer:
    def __init__(self):
        self.ai = AI(EpsilonGreedyStrategy(), 5)
        self.network = DeepQNetwork().to(DEVICE)
        self.target_network = DeepQNetwork().to(DEVICE)
        self.memory = ReplayMemory(replay_memory_capacity)

        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()  # This network will not be trained

        self.optimizer = optim.Adam(params=self.network.parameters(), lr=learning_rate)

    def get_action(self, state):
        return self.ai.select_action(
            torch.from_numpy(state).float(),
            self.network
        )

    def train(self):
        if not self.memory.can_provide_sample(batch_size):
            return

        experiences = self.memory.sample(batch_size)
        states, actions, next_states, rewards = extract_tensors(experiences)

        current_q_values = QValues.get_current(self.network, states, actions)
        next_q_values = QValues.get_next(self.target_network, next_states)
        target_q_values = (next_q_values * discount_rate) + rewards

        loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def set_result(self, state, action, reward, next_state):
        self.memory.push(Experience(
            torch.from_numpy(state).float(),
            action,
            torch.from_numpy(next_state).float(),
            reward
        ))

    def next_episode(self, number):
        if number % update_frequency == 0:
            self.target_network.load_state_dict(self.network.state_dict())


def get_path(num):
    return DATA_DIR + '/' + str(num) + '.' + EXTENSION


def read_ai_num(num):
    with open(get_path(num), 'rb') as f:
        return pickle.load(f)


def save_ai_num(ai, num):
    path = get_path(num)

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    elif os.path.exists(path):
        os.remove(path)

    with open(path, 'wb') as f:
        pickle.dump(ai, f)

################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
# This file contains the AI.
# Part of the code in this class is based on the DeepLizard
# tutorial about Deep Q Networks:
#     website: https://deeplizard.com
#
################################################################

from src.constants import *
from src.aiSettings import *

import random
import math
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple

# Defines the experiences as state, action, next state and reward
Experience = namedtuple(
    'Experience',  # Name of the tuple class
    ('state', 'action', 'next_state', 'reward')  # Content of the tuple
)


def map_to_input(map):
    """
    Get the input value as they will be given to the AI
    The input is an array of size COLUMNS * ROWS + 3 where the first COLUMNS * ROWS values
    represents the map. For each input, the value is -1 if the snake is at that position, 1 if there is food and
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
        inputs[piece[0] * COLUMNS + piece[1]] = -1

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


def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return t1, t2, t3, t4


class AI(object):
    """
    Represents the agent that will perform the actions based on a strategy (for instance an Epsilon Greedy Strategy)
    """

    def __init__(self, strategy, num_actions):
        """
        Constructor

        :param strategy: the strategy that will be used to choose whether to explore or not
        :param num_actions: the number of possible actions
        """
        self.strategy = strategy
        self.num_actions = num_actions
        self.current_step = 0

    def select_action(self, state, policy_network):
        """
        Selects an action to perform based on the strategy and the state

        :param state: the current state
        :param policy_network: the neural network that will be used if it chooses to act based on the experience
        :return: a number between 0 and self.num_actions that represents the chosen action
        """
        # Get the will to explore
        threshold = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if threshold > random.random():  # it chooses to explore
            # return random.randrange(self.num_actions)
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(DEVICE)
        else:  # it chooses to act on experience
            with torch.no_grad():
                # return policy_network(state).argmax(dim=1).to(DEVICE).item()
                return policy_network(state).argmax().to(DEVICE).item()
                # return policy_network(state).argmax().to(DEVICE)

    def calculate_loss(self, batch, neural_network, target_network):
        """
        Calculate MSE between actual state action values,
        and expected state action values from DQN
        """
        states, actions, rewards, dones, next_states = batch

        states_v = torch.from_numpy(states).float()
        next_states_v = torch.from_numpy(next_states).float()
        actions_v = torch.tensor(actions).to(DEVICE)
        rewards_v = torch.tensor(rewards).to(DEVICE)
        done = torch.ByteTensor(dones).to(DEVICE)

        # state_action_values = neural_network(states_v).gather(1, actions_v.long().unsqueeze(-1)).squeeze(-1)
        state_action_values = neural_network(states_v)
        # next_state_values = target_network(next_states_v).max(1)[0]
        next_state_values = target_network(next_states_v)
        # next_state_values[done] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * discount_rate + rewards_v
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def train(self, neural_network, replay_memory, target_network, optimizer):
        if not replay_memory.can_provide_sample(batch_size):
            return

        batch = replay_memory.sample(batch_size)

        for b in batch:
            optimizer.zero_grad()
            loss_t = self.calculate_loss(b, neural_network, target_network)
            loss_t.backward()
            optimizer.step()


class DeepQNetwork(nn.Module):
    """
    Deep Q Network with one input layer and one output layer. Each neuron is connected
        to all neurons of the next layer.

        The input layer has COLUMNS * ROWS + 3 inputs and 5 outputs
    """

    def __init__(self):
        """
        Constructor of the Deep Q Network
        """
        super().__init__()

        self.layer = nn.Linear(in_features=COLUMNS * ROWS + 3, out_features=5)

    def forward(self, t):
        # t = t.flatten(start_dim=1)  # Flatten the tensor t
        t = t.flatten()  # Flatten the tensor t

        return self.layer(t)  # Pass the tensor to the network


class ReplayMemory(object):
    """
    Contains all the past experiences that will be used to train the network
    """

    def __init__(self, capacity):
        """
        Constructor of the memory
        :param capacity: the number of experiences that will be stored
        """
        self.capacity = capacity  # The number of experiences that can be stored
        self.memory = []  # The experiences
        self.push_count = 0  # How many experiences have been stored

    def push(self, experience):
        """
        Add an experience to the memory
        :param experience: the experience to add
        """
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience

        self.push_count += 1

    def sample(self, batch_size):
        """
        Get a random sample of experiences from the memory
        :param batch_size: the number of experiences that will be returned
        :return: a random set of experiences
        """
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        """
        Get if there are enough experiences to complete a batch
        :param batch_size: the number of experiences that needs to be stored
        :return: True if it contains enough experiences, False otherwise
        """
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy(object):
    """
    Represents the will of the AI to explore new strategies or to play from experience
    """

    def __init__(self, start=max_exploration_rate, end=min_exploration_rate, decay=exploration_decay_rate):
        """
        Constructor

        :param start: the initial will to explore from 0 to 1
        :param end: the final will to explore from 0 to 1
        :param decay: the decay to pass from start to end from 0 to 1
        """
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        """
        Get the will to explore

        :param current_step: the current step
        :return: the will from 0 to 1
        """
        return self.end + (self.start - self.end) * math.exp(-1 * current_step * self.decay)


class QValues():
    @staticmethod
    def get_current(policy_network, states, actions):
        # return policy_network(states).gather(dim=1, index=actions.unsqueeze(-1))
        return policy_network(states).gather(index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_network, next_states):
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)

        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        # non_final_state_locations = [True] * len(next_states)
        # non_final_states = next_states

        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(DEVICE)
        values[non_final_state_locations] = target_network(non_final_states).max(dim=1)[0].detach()
        return values

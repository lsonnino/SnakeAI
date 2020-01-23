################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
# Based on 'Deep Q Learning is Simple with Tensorflow (Tutorial)' by Machine Learning with Phil tutorial on Youtube
#
################################################################

from src.aiSettings import *

import numpy as np
import tensorflow.compat.v1 as tf


tf.disable_v2_behavior()


class Network(object):
    def __init__(self, learning_rate, n_actions, name, input_dims, network_builder):
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.name = name
        self.input_dims = input_dims

        self.session = tf.Session()
        self.build_network(network_builder)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def build_network(self, network_builder):
        with tf.variable_scope(self.name):
            self.input, self.actions, self.q_target, self.Q_values = network_builder()

            self.loss = tf.reduce_mean(tf.square(self.Q_values - self.q_target))
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def load_checkpoint(self, file):
        self.saver.restore(self.session, file)

    def save_checkpoint(self, file):
        self.saver.save(self.session, file)

    def close(self):
        self.session.close()


class Agent(object):
    def __init__(self, n_actions, input_dims, name, network_builder):
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]

        self.q_eval = Network(learning_rate=learning_rate, n_actions=n_actions, name=name, input_dims=input_dims,
                              network_builder=network_builder)
        self.batch_size = batch_size

        self.gamma = discount_rate
        self.epsilon = max_exploration_rate
        self.epsilon_dec = exploration_decay_rate
        self.epsilon_end = min_exploration_rate

        self.memory_size = replay_memory_capacity
        self.state_memory = np.zeros( (self.memory_size, *input_dims) )
        self.new_state_memory = np.zeros( (self.memory_size, *input_dims) )
        self.action_memory = np.zeros( (self.memory_size, self.n_actions), dtype=np.int8 )
        self.reward_memory = np.zeros(self.memory_size)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.int8)
        self.memory_counter = 0

    def store_transition(self, state, chosen_action, reward, new_state, terminal):
        index = self.memory_counter % self.memory_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - terminal  # 0 if it is a terminal state

        actions = np.zeros(self.n_actions)
        actions[chosen_action] = 1.0
        self.action_memory[index] = actions

        self.memory_counter += 1

    def choose_action(self, state):
        # reshape the state because the input to the DQN is sized: none, *input_dims
        state = state[np.newaxis, :]

        if np.random.random() < self.epsilon:  # choose a random action
            action = np.random.choice(self.action_space)
        else:  # act based on neural network
            actions = self.q_eval.session.run(self.q_eval.Q_values, feed_dict={self.q_eval.input: state})
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory_counter < self.batch_size:  # not enough memory
            return

        # Get the batch
        max_mem = self.memory_counter if self.memory_counter < self.memory_size \
                                    else self.memory_size
        batch = np.random.choice(max_mem, self.batch_size)

        # Extract the batch
        state_batch = self.state_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        action_batch = self.action_memory[batch]
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action_batch, action_values)
        reward_batch = self.reward_memory[batch]
        terminal_batch = self.terminal_memory[batch]

        # Value of the current state
        q_eval = self.q_eval.session.run(self.q_eval.Q_values, feed_dict={self.q_eval.input: state_batch})
        # Target
        q_next = self.q_eval.session.run(self.q_eval.Q_values, feed_dict={self.q_eval.input: new_state_batch})

        # Get the loss but do not take consideration of the possible reward of a terminal states
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, action_indices] = reward_batch + self.gamma * np.max(q_next, axis=1) * terminal_batch

        # Train
        _ = self.q_eval.session.run(self.q_eval.train_op, feed_dict={
            self.q_eval.input: state_batch,
            self.q_eval.actions: action_batch,
            self.q_eval.q_target: q_target
        })

        # Update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_end)

    def load_models(self, file):
        self.q_eval.load_checkpoint(file)

    def save_models(self, file):
        self.q_eval.save_checkpoint(file)

    def close(self):
        self.q_eval.close()

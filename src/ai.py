################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
################################################################

from src.constants import *
from src.aiSettings import *

import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers


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

    index = np.argmax(outputs)

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

class AI:
    def __init__(self):
        self.batch_size = batch_size
        self.optimizer = optimizers.Adam(learning_rate)
        self.gamma = discount_rate
        self.number_of_actions = 5

        inputs = tf.keras.Input(shape=(COLUMNS * ROWS * 2,), name='input')
        x = tf.keras.layers.Dense(COLUMNS * ROWS, activation='tanh', name='hidden_layer')(inputs)
        outputs = tf.keras.layers.Dense(self.number_of_actions, activation='linear', name='output')(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name='SnakeAI')

        self.experience = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}
        self.max_experiences = replay_memory_capacity
        self.min_experiences = batch_size

    def predict(self, inputs):
        """
        :param inputs: can eather be a state or a batch of states
        """
        return self.model(np.atleast_2d(inputs.astype('float32')))

    @tf.function
    def train(self, TargetNet):
        # Do not train if has not acquired enough experience
        if len(self.experience['state']) < self.min_experiences:
            return 0

        # Select experiences from memory
        ids = np.random.randint(low=0, high=len(self.experience['state']), size=self.batch_size)

        # Extract experience
        states = np.asarray([self.experience['state'][i] for i in ids])
        actions = np.asarray([self.experience['action'][i] for i in ids])
        rewards = np.asarray([self.experience['reward'][i] for i in ids])
        next_states = np.asarray([self.experience['next_state'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])

        # Get predicted value  -- not well understood yet
        value_next = np.max(TargetNet.predict(next_states), axis=1)
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.model.number_of_actions), axis=1)
            loss = tf.math.reduce_sum(tf.square(actual_values - selected_action_values))

        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)

        # Apply back propagation
        self.optimizer.apply_gradients(zip(gradients, variables))

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            # Try something new
            rnd = np.random.choice(self.number_of_actions)
            output = np.zeros(self.number_of_actions)
            output[int(rnd)] = 1
            return get_output(output)
        else:
            # Use experience
            return get_output(self.predict(np.atleast_2d(states)))

    def add_experience(self, state, action, reward, next_state, done):
        exp = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}

        # If has gathered max experience, drop the oldest one
        if len(self.experience['state']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)

        # Add the experience to the memory
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        """
        Make this NN the same as {@code TrainNet}
        """
        '''
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables

        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
        '''
        self.model = tf.keras.models.clone_model(TrainNet.model)

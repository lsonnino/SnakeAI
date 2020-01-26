################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
# This file defines and explains each setting for the AI
#
################################################################

from src.models import *

# The bigger the discount rate, the more importance the AI will accord to future rewards
# in comparison with the present ones. It is often called Gamma
discount_rate = 0.8  # must be between 0 and 1

# The bigger the learning rate, the less the AI will use previous mistakes. It is often called Alpha
learning_rate = 0.0001  # must be between 0 and 1

# The number of experiences the AI will use to train after each step
batch_size = 128  # must be greater than 1
# The number of elements that can be contained in the memory
replay_memory_capacity = 65536  # must be greater than 1


# The maximum number the AI is allowed to make to get some food. If he gets
# food, he survives and reset its number. It dies otherwise.
AI_MAX_ALLOWED_MOVES = COLUMNS * ROWS  # must be less than 0 if infinite number of moves, greater than 0 otherwise


# Model selection

OMNISCIENT_MODEL = 0
TRI_DIRECTIONAL = 1
FOUR_DIRECTIONAL = 2
selected_model = FOUR_DIRECTIONAL  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
AI_NAME = 'Orochimaru'

if selected_model == TRI_DIRECTIONAL:
    AI_MODEL_BUILDER = tri_directional_ai_model_builder
    STATE_BUILDER = tri_directional_state_builder
    EMPTY_STATE_BUILDER = tri_directional_empty_state_builder
    INPUT_DIMENSION = [10]
elif selected_model == FOUR_DIRECTIONAL:
    AI_MODEL_BUILDER = four_directional_ai_model_builder
    STATE_BUILDER = four_directional_state_builder
    EMPTY_STATE_BUILDER = four_directional_empty_state_builder
    INPUT_DIMENSION = [7]
else:
    AI_MODEL_BUILDER = omniscient_ai_model_builder
    STATE_BUILDER = omniscient_state_builder
    EMPTY_STATE_BUILDER = omniscient_empty_state_builder
    INPUT_DIMENSION = [COLUMNS * ROWS + 4]

# Exploration rate

FULL_EXPLORATION_RATE_MODEL = 0
SMALL_EXPLORATION_RATE_MODEL = 1
NO_EXPLORATION_RATE_MODEL = 2
selected_exploration_rate_model = NO_EXPLORATION_RATE_MODEL  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

if selected_exploration_rate_model == FULL_EXPLORATION_RATE_MODEL:
    max_exploration_rate = 1  # must be between 0 and 1
    min_exploration_rate = 0.01  # must be between 0 and 1
elif selected_exploration_rate_model == SMALL_EXPLORATION_RATE_MODEL:
    max_exploration_rate = 0.5  # must be between 0 and 1
    min_exploration_rate = 0.001  # must be between 0 and 1
elif selected_exploration_rate_model == NO_EXPLORATION_RATE_MODEL:
    max_exploration_rate = 0  # must be between 0 and 1
    min_exploration_rate = 0  # must be between 0 and 1
exploration_decay_rate = 0.99999  # must be between 0 and 1

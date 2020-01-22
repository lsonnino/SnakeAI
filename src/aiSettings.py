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
learning_rate = 0.01  # must be between 0 and 1

max_exploration_rate = 0.99  # must be between 0 and 1
min_exploration_rate = 0.05  # must be between 0 and 1
exploration_decay_rate = 0.9999  # must be between 0 and 1
ss_exploration_decay_rate = 0.999  # must be between 0 and 1
ss_thresh = 0.7

# The number of experiences the AI will use to train after each step
batch_size = 128  # must be greater than 1
# The number of elements that can be contained in the memory
replay_memory_capacity = 1024  # must be greater than 1
# The number of times the AI will train before updating his semi-constant network
update_frequency = 100  # must be greater than 1


# The maximum number the AI is allowed to make to get some food. If he gets
# food, he survives and reset its number. It dies otherwise.
AI_MAX_ALLOWED_MOVES = 300  # must be less than 0 if infinite number of moves, greater than 0 otherwise

AI_MODEL_BUILDER = omniscient_ai_model_builder
STATE_BUILDER = omniscient_state_builder
EMPTY_STATE_BUILDER = omniscient_empty_state_builder

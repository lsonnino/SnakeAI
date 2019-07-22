################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
# This file defines and explains each setting for the AI
#
################################################################

# The bigger the discount rate, the more importance the AI will accord to future rewards
# in comparison with the present ones. It is often called Gamma
discount_rate = 0.6  # must be between 0 and 1

# The bigger the learning rate, the less the AI will use previous mistakes. It is often called Alpha
learning_rate = 0.5  # must be between 0 and 1

max_exploration_rate = 1  # must be between 0 and 1
min_exploration_rate = 0.1  # must be between 0 and 1
exploration_decay_rate = 0.001  # must be between 0 and 1


# The maximum number the AI is allowed to make to get some food. If he gets
# food, he survives and reset its number. It dies otherwise.
AI_MAX_ALLOWED_MOVES = 300  # must be less than 0 if infinite number of moves, greater than 0 otherwise

# The device to use for calculations
DEVICE = 'cpu'

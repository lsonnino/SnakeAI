from src.objects import *
from src.ai import *


class Game(object):
    def __init__(self, player, device, max_moves=-1):
        """
        Constructor of the game
        :param player: the player
        :param device: where to perform calculations
        :param max_moves: the maximum number of moves the snake is allowed to perform to get some food
                before dying. -1 means that there is no maximum.
        """
        self.max_moves = max_moves
        self.device = device
        self.player = player
        self.map = Map(max_moves=max_moves)
        self.map.spawn_food()

        self.playing = True

    def reset(self):
        """
        Reset the game to its initial state
        """
        self.map.__init__(max_moves=self.max_moves)
        self.map.spawn_food()

        self.playing = True

    def step(self):
        """
        Make a step in the game. Get the action to perform from the player and check the result
        :return: the reward the player got from its action
        """
        reward_val = 0

        # Get the player's action
        action = self.player.get_action(self.map)
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

        # Return the reward
        return torch.tensor([reward_val], device=self.device)

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


class AIPlayer:
    def __init__(self, device):
        self.ai = AI(EpsilonGreedyStrategy(), 5, device)
        self.network = DeepQNetwork()

    def get_action(self, map):
        return self.ai.select_action(
            torch.from_numpy(map_to_input(map)).float(),
            self.network
        )

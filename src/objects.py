################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
# This file contains every used object
#
################################################################

from src.constants import *

import random
import pygame
import numpy as np


def draw_case(window, color, x, y):
    """
    Draws a case
    :param window: the window to draw in
    :param color: the case's color
    :param x: the x position of the case
    :param y: the y position of the case
    """
    pygame.draw.rect(window, color, (CASE_WIDTH * x, CASE_HEIGHT * y, CASE_WIDTH, CASE_HEIGHT))


class Map(object):
    def __init__(self, max_moves=-1):
        """
        Instantiate the empty map with a newly created snake
        :param max_moves: the maximum number of moves the snake is allowed to perform to get some food
                before dying. -1 means that there is no maximum.
        """

        self.map = np.zeros((COLUMNS, ROWS), dtype=int)
        self.snake = Snake(max_moves)

    def draw(self, window):
        """
        Draws the map, the food and the snake
        :param window: the PyGame window object to draw in
        """

        # Runs the map
        for x in range(COLUMNS):
            for y in range(ROWS):
                if self.map[x, y] != EMPTY:
                    # Chooses the color
                    color = colors[self.map[x, y]]
                    # Draws the rectangle
                    draw_case(window, color, x, y)

        # Draws the snake
        self.snake.draw(window)

    def spawn_food(self):
        """
        Spawn a food case randomly in the map
        """

        found = False

        while not found:
            x = random.randint(0, COLUMNS - 1)
            y = random.randint(0, ROWS - 1)
            if not self.snake.body.__contains__((x, y)):
                self.map[x, y] = FOOD
                found = True

    def take_food(self, x, y):
        """
        Consumes the food at the given location
        :param x: the x location of the food to consume. Must be in range [0, ROWS[
        :param y: the y location of the food to consume. Must be in range [0, COLUMNS[
        """
        self.map[x, y] = EMPTY

    def check_food(self):
        """
        Check whether the snake took the food or not
        If so, consume the food and spawn another

        :return: True if some food has been taken, False otherwise
        """
        if self.map[self.snake.body[0]] == FOOD:
            # Remove the food
            self.take_food(self.snake.body[0][0], self.snake.body[0][1])
            # The snake took the food
            self.snake.took_food = True
            # Spawn some new food
            self.spawn_food()

            return True

        return False


class Snake(object):
    def __init__(self, max_moves=-1):
        """
        Initializes a sized 1 snake at the center of the screen looking to the right

        :param max_moves: the maximum number of moves the snake is allowed to perform to get some food
                before dying. -1 means that there is no maximum.
        """
        self.x = int(COLUMNS / 2)
        self.y = int(ROWS / 2)
        self.body = [(self.x, self.y)]

        self.direction = NONE
        self.took_food = False

        self.max_moves = max_moves
        self.moves_left = self.max_moves

    def draw(self, window):
        """
        Draws the snake in the given window
        :param window: the window to draw in
        """
        # Runs the body
        for piece in self.body:
            # Draws that piece
            draw_case(window, SNAKE_COLOR, piece[0], piece[1])

    def walk(self) -> bool:
        """
        Makes the snake walk one step taking in account whether he has just eaten or not and whether he is still alive
        afterwards
        :return: whether the snake is still alive after walking one step or not
        """
        if self.direction == RIGHT:  # Going rightwards
            new_head = (self.body[0][0] + 1, self.body[0][1])
        elif self.direction == LEFT:  # Going leftwards
            new_head = (self.body[0][0] - 1, self.body[0][1])
        elif self.direction == TOP:  # Going upwards
            new_head = (self.body[0][0], self.body[0][1] - 1)
        elif self.direction == BOTTOM:  # Going downwards
            new_head = (self.body[0][0], self.body[0][1] + 1)
        else:
            return True

        # Check the moves left
        if self.max_moves > 0:
            self.moves_left -= 1

            # If the snake has no more moves left
            if self.moves_left <= 0:
                return False

        if self.took_food:  # If the snake just took some food
            # Do not remove the last piece of the tail this turn
            self.took_food = False
        else:  # If not
            # Remove the last piece of the tail
            self.body.remove(self.body[len(self.body) - 1])

        # Check if he is still alive
        if self.body.__contains__(new_head):
            return False

        if new_head[0] < 0 or new_head[0] >= COLUMNS or new_head[1] < 0 or new_head[1] >= ROWS:
            return False

        # Move the head to new location
        self.body.insert(0, new_head)

        return True

    def got_food(self):
        self.moves_left = self.max_moves

    def get_score(self):
        """
        Get the snake's score
        :return: An integer representing the snake's score
        """
        if len(self.body) == 0:
            return 0

        return len(self.body) - 1

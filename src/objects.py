import random
import numpy as np
import pygame
from src.constants import *


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
    def __init__(self):
        """
        Instantiate the empty map with a newly created snake
        """

        self.map = np.zeros((ROWS, COLUMNS), dtype=int)
        self.snake = Snake()

    def draw(self, window):
        """
        Draws the map, the food and the snake
        :param window: the PyGame window object to draw in
        """

        # Runs the map
        for x in range(ROWS):
            for y in range(COLUMNS):
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
            x = random.randint(0, ROWS - 1)
            y = random.randint(0, COLUMNS - 1)
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
        """
        if self.map[self.snake.body[0]] == FOOD:
            # Remove the food
            self.take_food(self.snake.body[0][0], self.snake.body[0][1])
            # The snake took the food
            self.snake.took_food = True
            # Spawn some new food
            self.spawn_food()


class Snake(object):
    def __init__(self):
        """
        Initializes a sized 1 snake at the center of the screen looking to the right
        """
        self.x = int(ROWS / 2)
        self.y = int(COLUMNS / 2)
        self.body = [(self.x, self.y)]

        self.direction = NONE
        self.took_food = False

    def draw(self, window):
        """
        Draws the snnake in the given window
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
        elif self.direction == TOP:  # Going up
            new_head = (self.body[0][0], self.body[0][1] - 1)
        elif self.direction == BOTTOM:  # Going down
            new_head = (self.body[0][0], self.body[0][1] + 1)
        else:
            return True

        if self.took_food:  # If the snake just took some food
            # Do not remove the last piece of the tail this turn
            self.took_food = False
        else:  # If not
            # Remove the last piece of the tail
            self.body.remove(self.body[len(self.body) - 1])

        # Check if he is still alive
        if self.body.__contains__(new_head):
            return False

        if new_head[0] < 0 or new_head[0] >= ROWS or new_head[1] < 0 or new_head[1] >= COLUMNS:
            return False

        # Move the head to new location
        self.body.insert(0, new_head)

        return True

    def get_score(self):
        """
        Get the snake's score
        :return: An integer representing the snake's score
        """
        return len(self.body) - 1

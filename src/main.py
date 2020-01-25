################################################################
#
# Author: Lorenzo Sonnino
# GitHub: https://github.com/lsonnino
#
# The project:
#     The goal of this project is to make an AI that learns to play snake
#     using reinforcement learning -- by giving him a reward whenever he
#     does something good and by punishing him when he does something bad.
#
# Requirements:
#     * PyGame     : to display and play snake
#     * TensorFlow : manages the AI
#     * Numpy      : used to manage matrix
#
# This file is the main file -- this is the file that runs the game
# (the constants file indicates whether the AI plays or not)
#
################################################################

from src.game import *
import src.console as console
import matplotlib.pyplot as plt


def plot(scores, step_history, number_of_games):
    x = [i + 1 for i in range(number_of_games)]

    plt.figure()
    plt.title('Scores')
    plt.plot(x, scores)
    plt.savefig('scores.png')

    plt.figure()
    plt.title('Steps')
    plt.plot(x, step_history)
    plt.savefig('steps.png')

    plt.figure()
    plt.title('Optimization')
    combined = [scores[i] / step_history[i] for i in range(number_of_games)]
    plt.plot(x, combined)
    plt.savefig('optimization.png')


def on_exit(score_history, steps_history, game_num):
    if AI_PLAYS:
        save_ai_num(console.game.player, console.ai_generation - 1)
        console.game.player.brain.close()

        plot(score_history, steps_history, game_num - 1)


def main():
    score_history = []
    steps_history = []

    console.boot()
    game_num = 1  # keeps track of the number of played games

    try:
        # Keeps the game running
        running = True
        while running and (NUMBER_OF_GAMES < 0 or game_num <= NUMBER_OF_GAMES):
            # Reset the game
            console.game.reset()

            last_score, stopped = console.play()
            running = not stopped

            # Pass to next generation
            if AI_PLAYS:
                # Save the AI
                if game_num % SAVE_EVERY == 0:
                    save_ai_num(console.game.player, console.ai_generation)

                # Printing score
                print("AI score for gen " + str(console.ai_generation) + ": " + str(last_score))
                step = console.game.player.iteration
                print("current step: " + str(step) + " - greed: " + str(round(console.game.player.brain.epsilon * 100, 2)) + "%")
                score_history.append(last_score)
                steps_history.append(step)

                console.ai_generation += 1
            else:
                # Printing score
                print("Game score: " + str(last_score))

            game_num += 1

            console.game.next_episode()
    except KeyboardInterrupt:
        on_exit(score_history, steps_history, game_num)

    on_exit(score_history, steps_history, game_num)


snake_ai = compile('main()', 'snake_ai', 'exec')

exec(snake_ai)

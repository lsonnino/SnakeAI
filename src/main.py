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


def get_means(y, step=100):
    offset = int(step / 2)
    index = offset
    x = []
    means = []
    while index < len(y):
        x.append(index)
        means.append(np.mean(y[max(index - offset, 0) : min(index + offset, len(y))]))
        index += step

    return x, means


def plot_figure(x, y1, y2, axis, title):
    axis.plot(x, y1, 'bo')
    axis.set(title=title)

    if y2 is not None:
        ax2 = axis.twinx()
        ax2.plot(x, y2, 'xkcd:grey')


def plot(scores, step_history, epsilon_history, number_of_games):
    _, axis = plt.subplots(nrows=3, ncols=2, figsize=(30, 20))

    x = [i + 1 for i in range(number_of_games)]
    a, c = get_means(epsilon_history)

    _, b = get_means(scores)
    plot_figure(x, scores, epsilon_history, axis=axis[0, 0], title='Scores')
    plot_figure(a, b, c, axis=axis[0, 1], title='Scores means')

    _, b = get_means(step_history)
    plot_figure(x, step_history, epsilon_history, axis=axis[1, 0], title='Steps')
    plot_figure(a, b, c, axis=axis[1, 1], title='Steps means')

    combined = [scores[i] * np.mean(step_history) / step_history[i] for i in range(number_of_games)]
    _, b = get_means(combined)
    plot_figure(x, combined, epsilon_history, axis=axis[2, 0], title='Optimization')
    plot_figure(a, b, c, axis=axis[2, 1], title='Optimization means')

    plt.savefig('training_data.png', bbox_inches='tight', dpi=100)


def on_exit(score_history, steps_history, epsilon_history, game_num):
    if AI_PLAYS:
        save_ai_num(console.game.player, console.ai_generation - 1)
        console.game.player.brain.close()

        plot(score_history, steps_history, epsilon_history, game_num - 1)


def main():
    score_history = []
    steps_history = []
    epsilon_history = []

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
                epsilon_history.append(console.game.player.brain.epsilon * 100)

                console.ai_generation += 1
            else:
                # Printing score
                print("Game score: " + str(last_score))

            game_num += 1

            console.game.next_episode()
    except KeyboardInterrupt:
        on_exit(score_history, steps_history, epsilon_history, game_num)

    on_exit(score_history, steps_history, epsilon_history, game_num)


snake_ai = compile('main()', 'snake_ai', 'exec')

exec(snake_ai)

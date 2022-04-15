# Reference Links
# http://programarcadegames.com/index.php?lang=en&chapter=array_backed_grids
# https://stackoverflow.com/a/19120806

import time

import pandas as pd
from gym.envs.toy_text import frozen_lake
from mdptoolbox.example import forest
import numpy as np
import pygame
import matplotlib.pyplot as plt

seed = 42


def create_lake(size=8, p=0.85, random_state=seed):
    np.random.seed(random_state)

    frozen_map = frozen_lake.generate_random_map(size=size, p=p,)
    env = frozen_lake.FrozenLakeEnv(desc=frozen_map)

    num_actions = env.action_space.n
    num_states = env.observation_space.n
    R = np.zeros((num_actions, num_states, num_states))
    T = np.zeros((num_actions, num_states, num_states))

    for curr_state in np.arange(num_states):
        for action in np.arange(num_actions):
            for p, next_state, reward, done in env.P[curr_state][action]:
                T[action, curr_state, next_state] += p
                R[action, curr_state, next_state] = reward
    return T, R, frozen_map


class Simulator:
    def __init__(self, T, R, policy):
        super().__init__()

        self.state = None
        self.T = T
        self.R = R
        self.policy = policy

    def simulate(self, max_iters):
        state = 0
        rewards = []
        transitions = []
        iters = 1
        for _ in np.arange(max_iters):
            action = self.policy[state]
            next_state_probs = self.T[action][state]
            next_state = np.random.choice(np.arange(next_state_probs.size), p=next_state_probs)
            reward = 0
            if len(self.R.shape) == 3:
                reward = self.R[action][state][next_state]
            else:
                reward = self.R[state][action]
            rewards.append(reward)
            transitions.append((state, action, reward, next_state))
            if self.terminate(next_state):
                break
            state = next_state
            iters += 1
        return rewards[-1], np.sum(rewards), iters

    def terminate(self, state):
        raise NotImplementedError("done_check")


class FrozenLakeSimulator(Simulator):
    def __init__(self, T, R, policy, frozen_map):
        super().__init__(T, R, policy)
        self.frozen_map = frozen_map

    def terminate(self, state):
        rows = len(self.frozen_map)
        cols = len(self.frozen_map[0])
        row, col = np.unravel_index(state, (rows, cols))
        cell = self.frozen_map[row][col]
        return cell == 'H' or cell == 'G'


class TreeSimulator(Simulator):
    def __init__(self, T, R, policy):
        super().__init__(T, R, policy)

    def terminate(self, state):
        return state == 0


def render_lake(frozen_map, filename, policy=None):
    num_rows = len(frozen_map)
    num_columns = len(frozen_map[0])

    BLACK = (0, 0, 0)
    GREEN = (0, 200, 0)
    RED = (200, 0, 0)
    LIGHT_BLUE = (237, 227, 255)

    WIDTH = 20
    HEIGHT = 20
    MARGIN = 5

    pygame.init()
    WINDOW_SIZE = [((WIDTH + MARGIN) * num_columns) + MARGIN, ((HEIGHT + MARGIN) * num_rows) + MARGIN]
    screen = pygame.display.set_mode(WINDOW_SIZE)

    screen.fill(BLACK)

    for row in range(num_rows):
        for column in range(num_columns):
            map_value = frozen_map[row][column]
            if map_value == 'F':
                color = LIGHT_BLUE
            elif map_value == 'H':
                color = BLACK
            elif map_value == 'G':
                color = GREEN
            else:
                color = RED

            cell_rect = [(MARGIN + WIDTH) * column + MARGIN, (MARGIN + HEIGHT) * row + MARGIN, WIDTH, HEIGHT]
            pygame.draw.rect(screen, color, cell_rect)

            if policy is not None:
                policy_index = np.ravel_multi_index((row, column), (num_rows, num_columns))
                action_character = {
                    0: '<',
                    1: 'v',
                    2: '>',
                    3: '^'
                }[policy[policy_index]]
                p_val = pygame.font.Font(None, WIDTH).render(action_character, False, BLACK)
                rect = p_val.get_rect()
                rect.center = pygame.Rect(*cell_rect).center
                screen.blit(p_val, rect)

    pygame.display.flip()
    pygame.image.save(screen, filename)
    pygame.quit()


def create_forest(size=2500, wait_max_reward=5000, cut_max_reward=2500, p_fire=0.001):
    T, R = forest(S=size, r1=wait_max_reward, r2=cut_max_reward, p=p_fire)
    return T, R, np.empty(shape=(size,))


def render_forest(filename, policy, title):
    plt.clf()
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 2)
    for action in [1, 0]:
        x = np.argwhere(np.array(policy) == action)
        y = np.ones_like(x) * action
        ax.scatter(x=x, y=y, s=5, alpha=0.50)
    plt.yticks(np.arange(0, 2, 1))
    plt.xlabel("State")
    plt.ylabel("Action")
    plt.title(title)
    plt.legend(["Cut", "Wait"], loc="center", markerscale=2)
    plt.tight_layout()
    plt.savefig(filename)


def plot_forest(policy, ax):
    wait = None
    cut = None
    for action in [1, 0]:
        x = np.argwhere(np.array(policy) == action)
        y = np.ones_like(x) * action
        vals = ax.scatter(x=x, y=y, s=5, alpha=0.50)
        ax.set_yticks(np.arange(0, 2, 1))
        if action == 0:
            wait = vals
        else:
            cut = vals
        ax.legend(["Cut", "Wait"], loc="center", markerscale=2)

    plt.xlabel("State")
    plt.ylabel("Action")
    return wait, cut


def simulate_policy(simulator, epochs=100, max_iters=10_000):
    np.random.seed(seed)

    results = []
    for epoch in np.arange(epochs):
        start_time = time.time()
        reward, cumulative_reward, iters = simulator.simulate(max_iters=max_iters)
        end_time = time.time()
        time_taken = end_time - start_time
        results.append((epoch, time_taken, reward, cumulative_reward, iters))

    return pd.DataFrame(data=results, columns=['Epoch', 'Time', 'Reward', 'Cum Reward', 'Iterations'])

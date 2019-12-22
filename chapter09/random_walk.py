#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

########
# Start of tile3.py

"""
Tile Coding Software version 3.0beta
by Rich Sutton
based on a program created by Steph Schaeffer and others
External documentation and recommendations on the use of this code is available in the
reinforcement learning textbook by Sutton and Barto, and on the web.
These need to be understood before this code is.

This software is for Python 3 or more.

This is an implementation of grid-style tile codings, based originally on
the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm), but by now highly changed.
Here we provide a function, "tiles", that maps floating and integer
variables to a list of tiles, and a second function "tiles-wrap" that does the same while
wrapping some floats to provided widths (the lower wrap value is always 0).

The float variables will be gridded at unit intervals, so generalization
will be by approximately 1 in each direction, and any scaling will have
to be done externally before calling tiles.

Num-tilings should be a power of 2, e.g., 16. To make the offsetting work properly, it should
also be greater than or equal to four times the number of floats.

The first argument is either an index hash table of a given size (created by (make-iht size)),
an integer "size" (range of the indices from 0), or nil (for testing, indicating that the tile
coordinates are to be returned without being converted to indices).
"""

basehash = hash


class IHT:
    "Structure to handle collisions"

    def __init__(self, sizeval):
        self.size = sizeval
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count(self):
        return len(self.dictionary)

    def fullp(self):
        return len(self.dictionary) >= self.size

    def getindex(self, obj, readonly=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif readonly:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount == 0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count


def hashcoords(coordinates, m, readonly=False):
    if type(m) == IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m) == int: return basehash(tuple(coordinates)) % m
    if m == None: return coordinates


from math import floor, log
from itertools import zip_longest


def tiles(ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f * numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // numtilings)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles

def tileswrap(ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    qfloats = [floor(f * numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrapwidths):
            c = (q + b % numtilings) // numtilings
            coords.append(c % width if width else c)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles

# End of tile3.py
########

# the number of states except for terminal states
N_STATES = 1000

# all states, np.arange generate an array with the number from 1 to 1000
STATES = np.arange(1, N_STATES + 1)

# start from a central state
START_STATE = 500

# terminal states
END_STATES = [0, N_STATES + 1]

# possible actions
ACTION_LEFT = -1
ACTION_RIGHT = 1
ACTIONS = [ACTION_LEFT, ACTION_RIGHT]

# maximum stride for an action
STEP_RANGE = 100

def compute_true_value():
    # true state value, just a promising guess. true_value[0]=-1, true_value[1001]=1
    true_value = np.arange(-1001, 1003, 2) / 1001.0

    # Dynamic programming to find the true state values, based on the promising guess above
    # Assume all rewards are 0, given that we have already given value -1 and 1 to terminal states. TODO: is this ok?
    while True:
        old_value = np.copy(true_value)
        for state in STATES:
            true_value[state] = 0
            for action in ACTIONS:
                for step in range(1, STEP_RANGE + 1):
                    step *= action
                    next_state = state + step
                    # min() returns the minimum number in (). Here, the final return is next_state or 1001 (if
                    # next_state > 1001) or 0 (if next_state < 1001).
                    next_state = max(min(next_state, N_STATES + 1), 0)
                    # asynchronous update for faster convergence  TODO: why he says asynchronous update
                    true_value[state] += 1.0 / (2 * STEP_RANGE) * true_value[next_state]
        error = np.sum(np.abs(old_value - true_value))
        if error < 1e-2:
            break
    # correct the state value for terminal states to 0
    true_value[0] = true_value[-1] = 0

    return true_value

# take an @action at @state, return new state and reward for this transition
def step(state, action):
    step = np.random.randint(1, STEP_RANGE + 1)
    step *= action  # because action here only indicates to the position, left (-1) or right (1).
    state += step
    state = max(min(state, N_STATES + 1), 0)
    if state == 0:
        reward = -1
    elif state == N_STATES + 1:
        reward = 1
    else:
        reward = 0
    return state, reward

# get an action, following random policy
def get_action():
    if np.random.binomial(1, 0.5) == 1:
        return 1
    return -1

# a wrapper class for aggregation value function
class ValueFunction:
    # @num_of_groups: # of aggregations
    def __init__(self, num_of_groups):
        self.num_of_groups = num_of_groups
        self.group_size = N_STATES // num_of_groups

        # thetas
        self.params = np.zeros(num_of_groups)

    # get the value of @state
    def value(self, state):
        if state in END_STATES:
            return 0
        group_index = (state - 1) // self.group_size   # // is init division
        return self.params[group_index]

    # update parameters
    # @delta: step size * (target - old estimation)
    # @state: state of current sample
    def update(self, delta, state):
        group_index = (state - 1) // self.group_size
        self.params[group_index] += delta

# a wrapper class for tile coding value function
class TilingsValueFunction:
    # @num_of_tilings
    # @tileWidth: each tiling has several tiles, this parameter specifies the width of each tile
    # @tilingOffset: specifies how tilings are put together
    def __init__(self, numOfTilings, tileWidth, tilingOffset):
        self.numOfTilings = numOfTilings
        self.tileWidth = tileWidth
        self.tilingOffset = tilingOffset

        # To make sure that each sate is covered by same number of tiles,
        # we need one more tile for each tiling  TODO: why one more?
        # tilingSize is the number of tiles in one tiling.
        self.tilingSize = N_STATES // tileWidth + 1

        # weight for each tile
        self.params = np.zeros((self.numOfTilings, self.tilingSize))

        # For performance, only track the starting position for each tiling
        # As we have one more tile for each tiling, the starting position will be negative
        self.tilings = np.arange(-tileWidth + 1, 0, tilingOffset)   # np.arange(start,stop,step)

    # get the value of @state
    def value(self, state):
        stateValue = 0.0
        # go through all the tilings
        for tilingIndex in range(0, len(self.tilings)):
            # find the active tile in current tiling
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
            stateValue += self.params[tilingIndex, tileIndex]
        return stateValue

    # update parameters
    # @delta: step size * (target - old estimation)
    # @state: state of current sample
    def update(self, delta, state):

        # each state is covered by same number of tilings
        # so the delta should be divided equally into each tiling (tile)
        delta /= self.numOfTilings

        # go through all the tilings
        for tilingIndex in range(0, len(self.tilings)):
            # find the active tile in current tiling
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
            self.params[tilingIndex, tileIndex] += delta


# class for the value function by the tile coding code from Rich
# http://www.incompleteideas.net/tiles/tiles3.html
class Tiles3ValueFunction:
    # @num_of_tilings
    # @tileWidth: each tiling has several tiles, this parameter specifies the width of each tile
    # @tilingOffset: specifies how tilings are put together
    # def __init__(self, num_of_tilings, tileWidth, tilingOffset):
    def __init__(self, iht, num_tilings, floats, ints):


        # self.num_of_tilings = num_of_tilings
        self.iht = iht
        self.num_of_tilings = num_tilings
        self.floats = floats
        self.ints = ints
        # self.tileWidth = tileWidth
        self.tileWidth = 1
        # self.tilingOffset = tilingOffset
        self.tilingOffset = 1 / num_tilings
        # To make sure that each sate is covered by same number of tiles,
        # we need one more tile for each tiling  TODO: why one more? =》 in order to cover the whole state in each tiling
        # tilingSize is the number of tiles in one tiling.
        # self.tilingSize = N_STATES // tileWidth + 1
        self.num_of_tiles_per_tiling = self.iht // num_tilings + 1
        # weight for each tile
        self.params = np.zeros((self.num_of_tilings, self.num_of_tiles_per_tiling))

        # For performance, only track the starting position for each tiling  TODO：what is the meaning of self.tilings
        # As we have one more tile for each tiling, the starting position will be negative
        self.tilings = np.arange(-self.tileWidth + 1 / num_tilings, 0, self.tilingOffset)   # np.arange(start,stop,step)

    # # get the value of @state
    # def value(self, state):
    #     stateValue = 0.0
    #     # go through all the tilings
    #     for tilingIndex in range(0, self.num_of_tilings):
    #         # find the active tile in current tiling
    #         tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
    #         stateValue += self.params[tilingIndex, tileIndex]
    #     return stateValue

    # get the value of @state
    def value(self, state):
        stateValue = 0.0
        # state = self.floats
        # go through all the tilings
        for tiling_index in range(0, self.num_of_tilings):
            tile_indices = tiles(self.iht, self.num_of_tilings, state, self.ints)
            stateValue += self.params[tiling_index, tile_indices[tiling_index]]

    # update parameters
    # @delta: step size * (target - old estimation)
    # @state: state of current sample
    def update(self, delta, state):
        # state = self.floats
        # each state is covered by same number of tilings
        # so the delta should be divided equally into each tiling (tile)  TODO: what is delta
        delta /= self.num_of_tilings

        # go through all the tilings
        for tiling_index in range(0, self.num_of_tilings):
            # find the active tile in current tiling
            # tileIndex = (state - self.tilings[tiling_index]) // self.tileWidth
            tile_indices = tiles(self.iht, self.num_of_tilings, state, self.ints)
            self.params[tiling_index, tile_indices[tiling_index]] += delta

# a wrapper class for polynomial / Fourier -based value function
POLYNOMIAL_BASES = 0
FOURIER_BASES = 1
class BasesValueFunction:
    # @order: # of bases, each function also has one more constant parameter (called bias in machine learning)
    # @type: polynomial bases or Fourier bases
    def __init__(self, order, type):
        self.order = order
        self.weights = np.zeros(order + 1)

        # set up bases function
        self.bases = []
        if type == POLYNOMIAL_BASES:
            for i in range(0, order + 1):
                self.bases.append(lambda s, i=i: pow(s, i))   # x = pow(4, 3), Return the value of 4 to the power of 3 (same as 4 * 4 * 4)
        elif type == FOURIER_BASES:
            for i in range(0, order + 1):
                self.bases.append(lambda s, i=i: np.cos(i * np.pi * s))

    # get the value of @state
    def value(self, state):
        # map the state space into [0, 1]
        state /= float(N_STATES)
        # get the feature vector
        feature = np.asarray([func(state) for func in self.bases])
        return np.dot(self.weights, feature)

    def update(self, delta, state):
        # map the state space into [0, 1]
        state /= float(N_STATES)
        # get derivative value
        # @delta: step size * (target - old estimation)
        # derivative_value = np.asarray([func(state) for func in self.bases])
        # self.weights += delta * derivative_value
        # The following two lines code were changed from above.
        feature = np.asarray([func(state) for func in self.bases])
        self.weights += delta * feature


# gradient Monte Carlo algorithm
# @value_function: an instance of class ValueFunction
# @alpha: step size
# @distribution: array to store the distribution statistics
def gradient_monte_carlo(value_function, alpha, distribution=None):
    state = START_STATE
    trajectory = [state]

    # We assume gamma = 1, so return is just the same as the latest reward TODO: why? no right? maybe gamma = 0
    reward = 0.0
    while state not in END_STATES:
        action = get_action()
        next_state, reward = step(state, action)
        trajectory.append(next_state)
        state = next_state

    # Gradient update for each state in this trajectory
    for state in trajectory[:-1]:
        delta = alpha * (reward - value_function.value(state))
        value_function.update(delta, state)
        if distribution is not None:
            distribution[state] += 1

# semi-gradient n-step TD algorithm
# @valueFunction: an instance of class ValueFunction
# @n: # of steps
# @alpha: step size
def semi_gradient_temporal_difference(value_function, n, alpha):
    # initial starting state
    state = START_STATE

    # arrays to store states and rewards for an episode
    # space isn't a major consideration, so I didn't use the mod trick
    states = [state]
    rewards = [0]

    # track the time
    time = 0

    # the length of this episode
    T = float('inf')
    while True:
        # go to next time step
        time += 1

        if time < T:
            # choose an action randomly
            action = get_action()
            next_state, reward = step(state, action)

            # store new state and new reward
            states.append(next_state)
            rewards.append(reward)

            if next_state in END_STATES:
                T = time

        # get the time of the state to update
        update_time = time - n
        if update_time >= 0:
            returns = 0.0
            # calculate corresponding rewards
            for t in range(update_time + 1, min(T, update_time + n) + 1):
                returns += rewards[t]
            # add state value to the return
            if update_time + n <= T:
                returns += value_function.value(states[update_time + n])
            state_to_update = states[update_time]
            # update the value function
            if not state_to_update in END_STATES:
                delta = alpha * (returns - value_function.value(state_to_update))
                value_function.update(delta, state_to_update)
        if update_time == T - 1:
            break
        state = next_state
#%%
# Figure 9.1, gradient Monte Carlo algorithm
def figure_9_1(true_value):
    episodes = int(1e5)
    alpha = 2e-5

    # we have 10 aggregations in this example, each has 100 states
    value_function = ValueFunction(10)
    distribution = np.zeros(N_STATES + 2)
    for ep in tqdm(range(episodes)):
        gradient_monte_carlo(value_function, alpha, distribution)

    distribution /= np.sum(distribution)
    state_values = [value_function.value(i) for i in STATES]

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    plt.plot(STATES, state_values, label='Approximate MC value')
    plt.plot(STATES, true_value[1: -1], label='True value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(STATES, distribution[1: -1], label='State distribution')
    plt.xlabel('State')
    plt.ylabel('Distribution')
    plt.legend()

    plt.savefig('../images/figure_9_1.png')
    plt.close()

# semi-gradient TD on 1000-state random walk
def figure_9_2_left(true_value):
    episodes = int(1e5)
    alpha = 2e-4
    value_function = ValueFunction(10)
    for ep in tqdm(range(episodes)):
        semi_gradient_temporal_difference(value_function, 1, alpha)

    stateValues = [value_function.value(i) for i in STATES]
    plt.plot(STATES, stateValues, label='Approximate TD value')
    plt.plot(STATES, true_value[1: -1], label='True value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()
#%%
# different alphas and steps for semi-gradient TD
def figure_9_2_right(true_value):
    # all possible steps
    steps = np.power(2, np.arange(0, 10))

    # all possible alphas
    alphas = np.arange(0, 1.1, 0.1)

    # each run has 10 episodes
    episodes = 10

    # perform 100 independent runs
    runs = 100

    # track the errors for each (step, alpha) combination
    errors = np.zeros((len(steps), len(alphas)))
    for run in tqdm(range(runs)):
        for step_ind, step in zip(range(len(steps)), steps):
            for alpha_ind, alpha in zip(range(len(alphas)), alphas):
                # we have 20 aggregations in this example
                value_function = ValueFunction(20)
                for ep in range(0, episodes):
                    semi_gradient_temporal_difference(value_function, step, alpha)
                    # calculate the RMS error
                    state_value = np.asarray([value_function.value(i) for i in STATES])
                    errors[step_ind, alpha_ind] += np.sqrt(np.sum(np.power(state_value - true_value[1: -1], 2)) / N_STATES)
    # take average
    errors /= episodes * runs
    # truncate the error
    for i in range(len(steps)):
        plt.plot(alphas, errors[i, :], label='n = ' + str(steps[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.ylim([0.25, 0.55])
    plt.legend()

def figure_9_2(true_value):
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    figure_9_2_left(true_value)
    plt.subplot(2, 1, 2)
    figure_9_2_right(true_value)

    plt.savefig('../images/figure_9_2.png')
    plt.close()
#%%
# Figure 9.5, Fourier basis and polynomials
def figure_9_5(true_value):
    # my machine can only afford 1 run
    runs = 1

    episodes = 5000

    # # of bases
    orders = [5, 10, 20]

    alphas = [1e-4, 5e-5]
    labels = [['polynomial basis'] * 3, ['fourier basis'] * 3]

    # track errors for each episode
    errors = np.zeros((len(alphas), len(orders), episodes))
    for run in range(runs):
        for i in range(len(orders)):
            value_functions = [BasesValueFunction(orders[i], POLYNOMIAL_BASES), BasesValueFunction(orders[i], FOURIER_BASES)]
            for j in range(len(value_functions)):
                for episode in tqdm(range(episodes)):

                    # gradient Monte Carlo algorithm
                    gradient_monte_carlo(value_functions[j], alphas[j])

                    # get state values under current value function
                    state_values = [value_functions[j].value(state) for state in STATES]

                    # get the root-mean-squared error
                    errors[j, i, episode] += np.sqrt(np.mean(np.power(true_value[1: -1] - state_values, 2)))

    # average over independent runs
    errors /= runs

    for i in range(len(alphas)):
        for j in range(len(orders)):
            plt.plot(errors[i, j, :], label='%s order = %d' % (labels[i][j], orders[j]))
    plt.xlabel('Episodes')
    # The book plots RMSVE, which is RMSE weighted by a state distribution
    plt.ylabel('RMSE')
    plt.legend()

    plt.savefig('../images/figure_9_5.png')
    plt.close()

#%%
# Figure 9.10, it will take quite a while
def figure_9_10(true_value):

    # My machine can only afford one run, thus the curve isn't so smooth
    runs = 1

    # number of episodes
    episodes = 5000

    num_of_tilings = 50

    # each tile will cover 200 states
    tile_width = 200

    # how to put so many tilings  TODO: what is 4 means?
    tiling_offset = 4

    labels = ['tile coding (50 tilings)', 'state aggregation (one tiling)']

    # track errors for each episode
    errors = np.zeros((len(labels), episodes))
    for run in range(runs):
        # initialize value functions for multiple tilings and single tiling
        value_functions = [TilingsValueFunction(num_of_tilings, tile_width, tiling_offset),
                         ValueFunction(N_STATES // tile_width)]
        print(value_functions)
        for i in range(len(value_functions)):
            for episode in tqdm(range(episodes)):
                # I use a changing alpha according to the episode instead of a small fixed alpha
                # With a small fixed alpha, I don't think 5000 episodes is enough for so many
                # parameters in multiple tilings.
                # The asymptotic performance for single tiling stays unchanged under a changing alpha,
                # however the asymptotic performance for multiple tilings improves significantly
                alpha = 1.0 / (episode + 1)

                # gradient Monte Carlo algorithm
                gradient_monte_carlo(value_functions[i], alpha)

                # get state values under current value function
                state_values = [value_functions[i].value(state) for state in STATES]

                # get the root-mean-squared error
                errors[i][episode] += np.sqrt(np.mean(np.power(true_value[1: -1] - state_values, 2)))

    # average over independent runs
    errors /= runs

    for i in range(0, len(labels)):
        plt.plot(errors[i], label=labels[i])
    plt.xlabel('Episodes')
    # The book plots RMSVE, which is RMSE weighted by a state distribution
    plt.ylabel('RMSE')
    plt.legend()

    plt.savefig('../images/figure_9_10plus.png')
    plt.close()
#%%
# Figure 9.11, it will take quite a while, by the tile_coding code from Rich
def figure_9_11(true_value):

    # My machine can only afford one run, thus the curve isn't so smooth
    runs = 1

    # number of episodes
    episodes = 5000

    num_of_tilings = 50

    # each tile will cover 200 states
    tile_width = 200

    # how to put so many tilings  TODO: what is 4 means?
    tiling_offset = 4

    labels = ['tile coding (50 tilings)', 'state aggregation (one tiling)']
    iht = IHT(1024)
    num_tilings = 64
    floats = state
    ints = None
    # track errors for each episode
    errors = np.zeros((len(labels), episodes))
    for run in range(runs):
        # initialize value functions for multiple tilings and single tiling
        value_functions = [Tiles3ValueFunction(iht, num_tilings, floats, ints),
                         ValueFunction(N_STATES // tile_width)]
        for i in range(len(value_functions)):
            for episode in tqdm(range(episodes)):
                # I use a changing alpha according to the episode instead of a small fixed alpha
                # With a small fixed alpha, I don't think 5000 episodes is enough for so many
                # parameters in multiple tilings.
                # The asymptotic performance for single tiling stays unchanged under a changing alpha,
                # however the asymptotic performance for multiple tilings improves significantly
                alpha = 1.0 / (episode + 1)

                # gradient Monte Carlo algorithm
                gradient_monte_carlo(value_functions[i], alpha)

                # get state values under current value function
                state_values = [value_functions[i].value(state) for state in STATES]

                # get the root-mean-squared error
                errors[i][episode] += np.sqrt(np.mean(np.power(true_value[1: -1] - state_values, 2)))

    # average over independent runs
    errors /= runs

    for i in range(0, len(labels)):
        plt.plot(errors[i], label=labels[i])
    plt.xlabel('Episodes')
    # The book plots RMSVE, which is RMSE weighted by a state distribution
    plt.ylabel('RMSE')
    plt.legend()

    plt.savefig('../images/figure_9_10plus.png')
    plt.close()
    


if __name__ == '__main__':
    true_value = compute_true_value()

    figure_9_1(true_value)
    figure_9_2(true_value)
    figure_9_5(true_value)
    figure_9_10(true_value)

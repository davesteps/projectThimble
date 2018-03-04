import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from random import randint


class Market(gym.Env):

    def __init__(self, symbols, obs_window=130, max_steps=260, test_mode=False):

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0., high=1000., shape=(1, obs_window))#,dtype='float64')
        self.observation = None
        self.symbols = symbols
        self.max_steps = max_steps
        self.obs_window = obs_window
        self.test_mode = test_mode

        self._reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):

        if self.test_mode:
            self.start_index = self.symbols.shape[0] - 261
        else:
            self.start_index = randint(260,self.symbols.shape[0]-260)
        self.end_index = self.start_index + self.max_steps
        self.current_index = self.start_index
        self.position = 2
        self.current_price = self.symbols[self.current_index]
        self.start_price = self.current_price
        self.total_reward = 0
        self.closed_positions = 0

        self.new_observation()

        return self.observation

    def new_observation(self):
        self.observation = self.symbols[self.current_index - self.obs_window:self.current_index].as_matrix()

    def _step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """

        assert self.action_space.contains(action)

        self.current_index += 1
        self.old_price = self.current_price
        self.current_price = self.symbols[self.current_index]


        if self.position == 1 and action == 1:
            # print('hold open')
            reward = self.get_percent_change(self.old_price)
        elif self.position == 1 and action == 0:
            # print('close')
            self.position = 0
            reward = 0
            self.closed_positions += 1
        elif (self.position == 0 and action == 0) or (self.position == 2 and action == 0):
            # print('hold closed')
            self.position = 0
            reward = 0
        elif (self.position == 0 and action == 1) or (self.position == 2 and action == 1):
            # print('open')
            self.position = 1
            self.position_open_price = self.old_price
            reward = self.get_percent_change(self.old_price)

        self.episode_change = self.get_percent_change(self.start_price)
        self.total_reward += reward

        done = False
        if self.current_index == self.end_index or self.total_reward < -100:
            done = True

        self.new_observation()

        return self.observation, reward, done, {'TR':self.total_reward, 'TD':self.episode_change}

    def get_percent_change(self, start_price):
        return round(((self.current_price - start_price) / start_price) * 100,1)


    #
    # def render(self, mode='human',close=True):
    #     # pass
    #     # fld = '/Users/davesteps/Google Drive/pycharmProjects/keras_RL/'
    #     if 'images' not in os.listdir():
    #         os.mkdir('images')
    #     # for i in range(len(frames)):
    #     # plt.ioff()
    #     plt.imshow(self.observation)
    #     plt.savefig('images/' + str(self.step_count) + ".png")
    #


class MarketMultiple(gym.Env):

    def __init__(self, symbols, obs_window=130, max_steps=260, test_mode=False):

        self.action_space = spaces.Discrete(symbols.shape[1] + 1)
        self.observation_space = spaces.Box(low=0., high=1500., shape=(symbols.shape[1], obs_window))  # ,dtype='float64')
        self.observation = None
        self.nullaction = symbols.shape[1]
        self.symbols = symbols
        self.max_steps = max_steps
        self.obs_window = obs_window
        self.test_mode = test_mode

        self._reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):

        if self.test_mode:
            self.start_index = self.symbols.shape[0] - 261
        else:
            self.start_index = randint(260, self.symbols.shape[0] - 260)
        self.end_index = self.start_index + self.max_steps - 1
        self.current_index = self.start_index
        # position = randint(0, 2)
        # current_price = float(symbols.iloc[current_index, [position]])
        # open_price = current_price
        self.total_reward = 0
        # closed_positions = 0

        self.new_observation()

        return self.observation

    def new_observation(self):
        self.observation = self.symbols[self.current_index - self.obs_window:self.current_index].as_matrix().transpose()

    def _step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """

        assert self.action_space.contains(action)
        self.current_index += 1

        if action == self.nullaction:
            reward = 0
        else:
            self.current_price = float(self.symbols.iloc[self.current_index, [action]])
            self.tomorrow_price = float(self.symbols.iloc[self.current_index + 1, [action]])
            reward = self.get_percent_change(self.current_price,self.tomorrow_price)
            self.total_reward += reward

        done = False
        if self.current_index == self.end_index or self.total_reward < -100:
            done = True

        self.baseline_change()

        self.new_observation()

        return self.observation, reward, done, {'TR':self.total_reward,
                                                'chg_mn':self.min_change,
                                                'chg_mx': self.max_change
                                                }

    def baseline_change(self):
        ii = [i for i in range(self.symbols.shape[1])]
        sp = self.symbols.iloc[self.start_index, ii].as_matrix()
        cp = self.symbols.iloc[self.current_index, ii].as_matrix()

        change = ((cp - sp) / sp) * 100
        self.min_change = round(min(change),1)
        self.max_change = round(max(change),1)



    def get_percent_change(self, p1, p2):
        return round(((p2 - p1) / p1) * 100,1)


    #
    # def render(self, mode='human',close=True):
    #     # pass
    #     # fld = '/Users/davesteps/Google Drive/pycharmProjects/keras_RL/'
    #     if 'images' not in os.listdir():
    #         os.mkdir('images')
    #     # for i in range(len(frames)):
    #     # plt.ioff()
    #     plt.imshow(self.observation)
    #     plt.savefig('images/' + str(self.step_count) + ".png")
    #


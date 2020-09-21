from numpy.random import randint
import pandas as pd
import numpy as np
from numba import jit
import gym
from gym import spaces


@jit('Tuple((i8, f8, b1))(f8, f8, f8[:], f8[:], f8[:], f8[:], i8, i8)', nopython=True)
def walk_forward(entry_pr, action_val, opn, close, high, low, cursor, n):
    direction = np.sign(action_val)
    while True:
        cursor += 1
        if cursor >= n:
            return n - 1, -action_val, True
        if direction == 1:
            if (entry_pr - low[cursor]) >= action_val:
                return cursor, -action_val, False
            if (high[cursor] - entry_pr) >= action_val:
                return cursor, action_val, False
        if direction == -1:
            if (high[cursor] - entry_pr) >= np.abs(action_val):
                return cursor, action_val, False
            if (entry_pr - low[cursor]) >= np.abs(action_val):
                return cursor, -action_val, False

# we will start simple with an assumed position size of 1 always but will consider enhancements in future
# we will also have equal profit targets and stop losses
class Environment(gym.Env):

    def __init__(self, df, features, actions=[0.0, 3.0, 5.0, 10.0], min_obs=5, add_features=0):
        super(Environment, self).__init__()
        self.df = df
        self.df = self.df.sort_values('time')
        self.day_df = None
        self.date = df.date.iloc[0]
        self.unique_dates = self.df.date.unique()
        self.min_obs = min_obs
        self.cursor = self.min_obs
        self.features = features
        self.actions = actions
        self.info = {'episode': {'r': 0}}
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(features) + add_features,),
            dtype=np.float64
        )

    def set_date(self, date):
        self.date = date

    def set_random_date(slef):
        self.date = self.unique_dates(randint(0, self.unique_dates.shape[0]))

    def reset(self):
        self.cursor = self.min_obs
        self.day_df = self.df.loc[self.df.date == self.date]
        return self.day_df.iloc[self.cursor][self.features]


    def step(self, action):
        action_val = self.actions[action]
        self.info['episode']['r'] = 0
        if (self.cursor + 1) >= self.day_df.shape[0]:
            return self.day_df.iloc[-1][self.features], 0, True, self.info['episode']['r']

        if action_val == 0:
            self.cursor += 1
            state = self.day_df.iloc[self.cursor][self.features]
            return state, 0, False, self.info['episode']['r']

        # In live trading the bars will be built in realtime
        # So we will assume it's possible to get a fill at the closing price
        # of the bar.
        close = self.day_df.iloc[self.cursor].close
        direction = np.sign(action_val)
        cursor, action_val, done = walk_forward(
            close, action_val, self.day_df.open.values,
            self.day_df.close.values, self.day_df.high.values,
            self.day_df.low.values, self.cursor, self.day_df.shape[0]
        )
        self.cursor = cursor
        self.info['episode']['r'] = action_val
        return self.day_df.iloc[self.cursor][self.features], action_val, done, self.info['episode']['r']

    def render(self):
        print('not implemented. Sorry.')

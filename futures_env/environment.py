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
        if direction == 1:
            if (entry_pr - low[cursor]) >= action_val:
                return cursor, -action_val, False
            if (high[cursor] - entry_pr) >= action_val:
                return cursor, action_val, False
            if cursor >= n:
                n - 1, (opn[n-1] - entry_pr), True # taking earliest exit on last bar
        if direction == -1:
            if (high[cursor] - entry_pr) >= np.abs(action_val):
                return cursor, action_val, False
            if (entry_pr - low[cursor]) >= np.abs(action_val):
                return cursor, -action_val, False
            if cursor >= n:
                n - 1, (entry_pr - opn[n-1]), True # taking earliest exit on last bar


@jit('Tuple((i8, f8, b1))(f8, f8, f8[:], f8[:], f8[:], f8[:], i8, i8)', nopython=True)
def walk_forward2(entry_pr, action_val, opn, close, high, low, cursor, n):
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

@jit('Tuple((i8, f8, b1))(f8, f8, f8[:], f8[:], f8[:], f8[:], i8, i8)', nopython=True)
def walk_forward3(entry_pr, action_val, opn, close, high, low, cursor, n):
    direction = np.sign(action_val)
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
    return cursor, 0, False

# we will start simple with an assumed position size of 1 always but will consider enhancements in future
# we will also have equal profit targets and stop losses
class Environment(gym.Env):

    def __init__(self, df, features, meta_cols, actions=[0.0, 3.0, 5.0, 10.0],
                 min_obs=5, add_features=0, skip_state=True, process_feats=True):
        super(Environment, self).__init__()
        self.df = df
        self.df = self.df.sort_values('time')
        self.day_df = None
        self.date = df.date.iloc[0]
        self.unique_dates = self.df.date.unique()
        self.min_obs = min_obs
        self.cursor = self.min_obs
        self.features = features
        self.meta_cols =meta_cols
        self.actions = actions
        self.total_rewards = 0
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(features) + add_features,),
            dtype=np.float32
        )
        self.skip_state = skip_state
        self.trade_terminal_state = None
        if process_feats:
            self.process_features()

    def process_features(self):
        df = self.df[self.features[:-1] + self.meta_cols] # excludind secs from features
        df.loc[:, 'rsi'] = df.rsi.fillna(50)
        df = df.replace([np.nan, np.inf, -np.inf], 0)
        df.loc[:, 'secs'] = ((df.time.dt.hour * 3600) + (df.time.dt.minute * 60) + (df.time.dt.second))
        sec_min, sec_max = df.secs.min(), df.secs.max()
        df.loc[:, 'secs'] -= sec_min
        df.loc[:, 'secs'] /= (sec_max - sec_min)
        #df.loc[:, ['rsi', 'sto']] /= 100
        mn = df[['rsi', 'sto', 'mv_std', 'mean_dist', 'std_frac', 'close_diff']].mean().values
        std = df[['rsi', 'sto', 'mv_std', 'mean_dist', 'std_frac', 'close_diff']].std().values
        df.loc[:, ['rsi', 'sto', 'mv_std', 'mean_dist', 'std_frac', 'close_diff']] -= std
        df.loc[:, ['rsi', 'sto', 'mv_std', 'mean_dist', 'std_frac', 'close_diff']] /= std
        self.df = df

    def set_date(self, date):
        self.date = date

    def set_random_date(slef):
        self.date = self.unique_dates(randint(0, self.unique_dates.shape[0]))

    def reset(self):
        self.cursor = self.min_obs
        self.day_df = self.df.loc[self.df.date == self.date]
        self.total_rewards = 0
        return self.day_df.iloc[self.cursor][self.features].values.astype(np.float32)


    def step(self, action):
        action_val = self.actions[action]
        if (self.cursor + 1) >= self.day_df.shape[0]:
            state, reward, done = self.day_df.iloc[-1][self.features].values.astype(np.float32), 0, True
        elif action_val == 0:
            self.cursor += 1
            state = self.day_df.iloc[self.cursor][self.features].values.astype(np.float32)
            state, reward, done = state, 0, False
        else:
            # In live trading the bars will be built in realtime
            # So we will assume it's possible to get a fill at the closing price
            # of the bar.
            cursor_start = self.cursor
            close = self.day_df.iloc[self.cursor].close

            cursor, action_val, done = walk_forward2(
                close, action_val, self.day_df.open.values,
                self.day_df.close.values, self.day_df.high.values,
                self.day_df.low.values, self.cursor, self.day_df.shape[0]
            )
            self.cursor = cursor if self.skip_state else (cursor_start + 1)
            state, reward = self.day_df.iloc[self.cursor][self.features].values.astype(np.float32), action_val
            self.trade_terminal_state = self.day_df.iloc[cursor][self.features].values.astype(np.float32)

            if (self.cursor + 1) >= self.day_df.shape[0]:
                done = True
            else:
                done = False
        self.total_rewards += reward
        if done:
            info = {'episode': {'r': self.total_rewards}}
            self.total_rewards = 0
        else:
            info = {}
        return state, reward, done, info

    def render(self):
        print('not implemented. Sorry.')



class EnvironmentNoSkip(Environment):

    def __init__(self, df, features, meta_cols, actions=[0.0, 3.0, 5.0, 10.0],
                 min_obs=5, add_features=0, skip_state=True,
                 process_feats=True):
        super(EnvironmentNoSkip, self).__init__(df, features, meta_cols,
                                                actions=actions,
                                                min_obs=min_obs,
                                                add_features=add_features,
                                                skip_state=True,
                                                process_feats=process_feats)
        self.in_trade = False
        self.trade_action = 0
        self.trade_entry = None
        self.trade_entries = []
        self.trade_exits = []

    def step(self, action):
        action_val = self.actions[action]
        if ((self.cursor + 1) >= self.day_df.shape[0]) and (not self.in_trade):
            state, reward, done = self.day_df.iloc[-1][self.features].values.astype(np.float32), 0, True
        elif (action_val == 0) and (not self.in_trade):
            self.cursor += 1
            state = self.day_df.iloc[self.cursor][self.features].values.astype(np.float32)
            state, reward, done = state, 0, False
        else:
            # In live trading the bars will be built in realtime
            # So we will assume it's possible to get a fill at the closing price
            # of the bar.
            close = self.day_df.iloc[self.cursor].close
            if not self.in_trade:
                self.in_trade = True
                self.trade_action = action_val
                self.trade_entry = close
                self.trade_entries.append(self.day_df.iloc[self.cursor][self.features].values.astype(np.float32)[-1])

            cursor, action_val, done = walk_forward3(
                self.trade_entry, self.trade_action, self.day_df.open.values,
                self.day_df.close.values, self.day_df.high.values,
                self.day_df.low.values, self.cursor, self.day_df.shape[0]
            )
            if action_val != 0: # only non zero returns indicate the end of a trade
                self.in_trade = False
                self.trade_exits.append(self.day_df.iloc[cursor][self.features].values.astype(np.float32)[-1])
            self.cursor = cursor
            state, reward = self.day_df.iloc[self.cursor][self.features].values.astype(np.float32), action_val
        self.total_rewards += reward
        if done:
            info = {'episode': {'r': self.total_rewards}}
            self.total_rewards = 0
            self.trade_entries = []
            self.trade_exits = []
        else:
            info = {}
        return state, reward, done, info


class EnvSkipState(Environment):

    def __init__(self):
        df = pd.read_parquet('/Users/brianmcclanahan/git_repos/AllAboutFuturesRL/historical_index_data/S_and_P_historical.parquet')
        feature_cols = ['mv_std', 'mean_dist', 'std_frac', 'sto', 'rsi', 'close_diff', 'secs']
        meta_cols = ['open', 'high', 'low', 'close', 'mv_avg', 'date', 'time']
        super(EnvSkipState, self).__init__(df, feature_cols, meta_cols, actions=[0.0, 3.0, 5.0, 10.0], min_obs=5, add_features=0)
        super(EnvSkipState, self).set_date(self.unique_dates[1])


class EnvFull(Environment):

    def __init__(self):
        df = pd.read_parquet('/Users/brianmcclanahan/git_repos/AllAboutFuturesRL/historical_index_data/S_and_P_historical.parquet')
        feature_cols = ['mv_std', 'mean_dist', 'std_frac', 'sto', 'rsi', 'close_diff', 'secs']
        meta_cols = ['open', 'high', 'low', 'close', 'mv_avg', 'date', 'time']
        super(EnvFull, self).__init__(df, feature_cols, meta_cols, actions=[0.0, 3.0, 5.0, 10.0], min_obs=5, add_features=0,
                                      skip_state=False)
        super(EnvFull, self).set_date(self.unique_dates[1])

# figure out how to parameterize these environments
class EnvFullV2(EnvironmentNoSkip):

    def __init__(self):
        df = pd.read_parquet('/Users/brianmcclanahan/git_repos/AllAboutFuturesRL/historical_index_data/S_and_P_historical.parquet')
        feature_cols = ['mv_std', 'mean_dist', 'std_frac', 'sto', 'rsi', 'close_diff', 'secs']
        meta_cols = ['open', 'high', 'low', 'close', 'mv_avg', 'date', 'time']
        super(EnvFullV2, self).__init__(df, feature_cols, meta_cols, actions=[0.0, 3.0, 5.0, 10.0], min_obs=5, add_features=0,
                                       skip_state=False)
        super(EnvFullV2, self).set_date(self.unique_dates[1])

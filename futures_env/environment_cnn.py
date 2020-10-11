from numpy.random import randint
import pandas as pd
import numpy as np
from numba import jit
import gym
from gym import spaces
import datetime
from datetime import timedelta
from .environment import Environment


# we will start simple with an assumed position size of 1 always but will consider enhancements in future
# we will also have equal profit targets and stop losses
class CNNEnvironment(Environment):

    def __init__(self, df, diff_features, meta_cols, scaler_features=[],
                 actions=[0.0, 2.0, 3.0, 5.0, 10.0],
                 window_len=360, min_obs=5, skip_state=True,
                 normalize_feats=False, low=-10, high=10, process_feats=True,
                 random_samp=False, trade_start_secs=int(3600 * 9.5)):
        self.scaler_features = scaler_features
        self.diff_features = [f'{feat}_diff' for feat in diff_features]
        self.all_features = self.diff_features + scaler_features
        self.normalize_feats = normalize_feats
        super(CNNEnvironment, self).__init__(
            df, diff_features, meta_cols,
            actions=actions,
            min_obs=min_obs,
            add_features=0,
            process_feats=process_feats,
            random_samp=random_samp,
            normalize_feats=self.normalize_feats,
            skip_state=skip_state
        )
        self.trade_start_secs = trade_start_secs
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.all_features), window_len),
            dtype=np.float32
        )
        self.unique_dates = sorted(self.unique_dates)[1:]
        self.window_len = window_len
        self.mean = 0

    def process_features(self):
        df = self.df
        self.df.loc[:, 'secs'] = ((df.time.dt.hour * 3600) + (df.time.dt.minute * 60) + (df.time.dt.second))
        for feat in self.features:
            self.df.loc[:, feat + '_diff'] = self.df[feat].diff()
        self.df.loc[:, 'last_date'] = self.df.date.shift()
        self.df.loc[self.df.last_date != self.df.date, self.diff_features] = 0

    def normalize_features(self):
        df = self.df
        avg = df[self.all_features].mean().values
        std = df[self.all_features].std().values
        self.df.loc[:, self.all_features] -= avg
        self.df.loc[:, self.all_features] /= std

    def reset(self):
        self.cursor = self.min_obs
        if self.random_samp:
            index = np.random.randint(0, len(self.unique_dates))
            self.date = self.unique_dates[index]
        date_positions = np.where(self.df.date == self.date)[0]
        first_date_index = date_positions[0]
        last_date_index = date_positions[-1]
        self.day_df = self.df.iloc[min(0, first_date_index - self.window_len) + 1:last_date_index + 1]
        first_index = np.where(
            (self.day_df.date == self.date) & (self.day_df.secs >= self.trade_start_secs)
        )[0][0]
        self.cursor = first_index
        self.total_rewards = 0
        # features in each channel are differences. Set them to zero at the start of the day
        #self.day_df.loc[self.day_df.index[self.cursor], self.diff_features] = self.mean
        state = self.day_df.iloc[self.cursor - self.window_len + 1:self.cursor + 1][self.all_features].T.values.astype(np.float32)
        #print('restate ', state.shape, self.day_df.shape, self.cursor, self.date)
        return state

    def step(self, action):
        action_val = self.actions[action]
        state, reward, done, info = super(CNNEnvironment, self).step_w_action(action_val)
        state = self.day_df.iloc[self.cursor - self.window_len + 1:self.cursor + 1][self.all_features].T.values.astype(np.float32)
        #print('state ', state.shape, self.cursor, self.day_df.shape, done, reward)
        return state, reward, done, info


class EnvCNNSkipState(CNNEnvironment):

    def __init__(self, df=None, set_date=True): # set date is just here for compatibility
        if df is None:
            df = pd.read_parquet('/Users/brianmcclanahan/git_repos/pytorch-a2c-ppo-acktr-gail/datasets/S_and_P_train2.parquet')
            #df = df.loc[df.time.between(datetime.datetime(2010, 1, 1), datetime.datetime(2013, 1, 1))]
        #feature_cols = ['mv_std', 'std_frac', 'sto', 'rsi', 'secs'] 'rsi', 'adx'
        feature_cols = ['close', 'open']
        scaler_features = [] #['volume']
        meta_cols = ['open', 'high', 'low', 'close', 'date', 'time']
        super(EnvCNNSkipState, self).__init__(
            df, feature_cols, meta_cols, actions=[-10, -5.0, -3,0, -2.0, 0.0, 2.0, 3.0, 5.0, 10.0],
            min_obs=5, random_samp=(not set_date), window_len=16,
            scaler_features=scaler_features, normalize_feats=False)
        if set_date:
            super(EnvCNNSkipState, self).set_date(self.unique_dates[1])

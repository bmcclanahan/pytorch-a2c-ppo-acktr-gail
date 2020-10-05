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
                 window_len=360, min_obs=5, add_features=0, skip_state=True,
                 normalize_feats=False, low=-10, high=10, process_feats=True,
                 random_samp=False, trade_start_secs=int(3600 * 9.5)):
        super(Environment, self).__init__(
            df, diff_features, meta_cols,
            actions=actions,
            min_obs=min_obs,
            add_features=add_features,
            process_feats=process_feats,
            random_samp=random_samp,
            normalize_feats=normalize_feats,
            skip_state=skip_state
        )
        self.scaler_features = scaler_features
        self.diff_features = [f'{feat}_diff' for feat in diff_features]
        self.all_features = self.diff_features + scaler_features
        self.trade_start_secs = trade_start_secs
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(features) + add_features, window_len),
            dtype=np.float32
        )
        self.unique_dates = sorted(self.unique_dates)[1:]
        self.window_len = window_len


    def process_features(self):
        for feat in self.features:
            self.df.loc[:, feat + '_diff'] = self.df[feat].diff()

    def reset(self):
        self.cursor = self.min_obs
        if self.random_samp:
            index = np.random.randint(0, self.unique_dates.shape[0])
            self.date = self.unique_dates[index]
        self.day_df = self.df.loc[self.df.date.between(self.date - timedelta(days=1), self.date)]
        first_index = np.where(
            (self.day_df == self.date) & (self.day_df.secs > self.trade_start_secs)
        )[0][0]
        self.cursor = first_index
        self.total_rewards = 0
        # features in each channel are differences. Set them to zero at the start of the day
        self.day_df.loc[self.day_df.index[self.cursor], self.diff_features] = 0
        return self.day_df.iloc[self.cursor - self.window_len:self.cursor][self.all_features].values.astype(np.float32)

    def step(self, action):
        action_val = self.actions[action]
        state, reward, done, info = super(Environment, self).step_w_action(action_val)
        state = self.day_df.iloc[self.cursor - self.window_len:self.cursor][self.all_features].values.astype(np.float32)
        return state, reward, done, info


class EnvCNNSkipState(CNNEnvironment):

    def __init__(self, df=None, set_date=True): # set date is just here for compatibility
        if df is None:
            df = pd.read_parquet('/Users/brianmcclanahan/git_repos/AllAboutFuturesRL/historical_index_data/S_and_P_historical.parquet')
            df = df.loc[df.time.between(datetime.datetime(2010, 1, 1), datetime.datetime(2013, 1, 1))]
        #feature_cols = ['mv_std', 'std_frac', 'sto', 'rsi', 'secs'] 'rsi', 'adx'
        feature_cols = ['pr_diff_ewa', 'volume_roc', 'sto', 'rsi', 'adx', 'secs']
        meta_cols = ['open', 'high', 'low', 'close', 'date', 'time']
        super(EnvSkipStateTraining, self).__init__(df, feature_cols, meta_cols, actions=[-10, -5.0, -3,0, -2.0, 0.0, 2.0, 3.0, 5.0, 10.0], min_obs=5, add_features=0,
                                                   random_samp=True)

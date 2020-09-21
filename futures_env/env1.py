from .base_environment import Environment
import pandas as pd


# we will start simple with an assumed position size of 1 always but will consider enhancements in future
# we will also have equal profit targets and stop losses
class Environment(Environment):

    def __init__(self):
        df = pd.read_parquet('/Users/brianmcclanahan/git_repos/AllAboutFuturesRL/historical_index_data/S_and_P_historical.parquet')
        feature_cols = ['mv_std', 'mean_dist', 'std_frac', 'sto', 'rsi', 'close_diff', 'secs']
        super(Environment, self).__init__(df, features, actions=[0.0, 3.0, 5.0, 10.0], min_obs=5, add_features=0)
        super(Environment, self).set_date(super(Environment, self).unique_dates[1])

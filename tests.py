from futures_env.environment import EnvironmentNoSkip, Environment
import pandas as pd
import numpy as np
import datetime

def build_test_df(price):
    df = pd.DataFrame({col:price for col in ['open', 'high', 'low', 'close']})
    df.loc[:, 'secs'] = np.arange(df.shape[0])
    df.loc[:, 'time'] = np.arange(df.shape[0])
    df = df.astype(np.float64)
    df.loc[:, 'date'] = datetime.datetime(2020, 1, 1)
    return df

def gather_rewards_and_states(env, action):
    state = env.reset()
    rewards = []
    states = []
    done = False
    counter = 0 # using a counter for indexing instead of a for loop to test that done is being returned correctly
    while not done:
        state, reward, done, info = env.step(action[counter])
        counter += 1
        rewards.append(reward)
        states.append(state)
    return rewards, states


def test_env_no_skip():
    features = ['secs']
    meta_cols = ['open', 'high', 'low', 'close']

    #scenario 1
    price =  np.array([1, 2, 3, 3, 4, 5, 6, 9, 8, 7, 9, 5, 4])
    df = build_test_df(price)
    env = EnvironmentNoSkip(df, features, meta_cols, actions=[0, 1, 2], min_obs=0,
                            add_features=0, skip_state=False, process_feats=False)
    action =  np.array([0, 2, 0, 1, 2, 2, 0, 1, 0, 1, 0, 0, 0])
    expected_rewards = np.array([0, 0, 0, 2, 0, 2, 0, -1, 0, 1, 0, 0, 0])
    rewards, states = gather_rewards_and_states(env, action)
    assert np.array_equal(np.array(rewards), np.array(expected_rewards).astype(np.float64))
    assert np.array_equal(
        np.array(states).flatten(),
        np.array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 12.])
    )

    #scenario 2
    price =  np.array([1, 2, 3, 3, 4, 5, 6, 9, 8, 7, 6, 5, 5])
    action = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    df = build_test_df(price)
    env = EnvironmentNoSkip(df, features, meta_cols, actions=[0, 4], min_obs=0,
                            add_features=0, skip_state=False, process_feats=False)
    rewards, states = gather_rewards_and_states(env, action)
    expected_rewards = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4])
    assert np.array_equal(np.array(rewards), np.array(expected_rewards).astype(np.float64))
    assert np.array_equal(
        np.array(states).flatten(),
        np.array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 12.])
    )

def test_env_trade_everything():
    features = ['secs']
    meta_cols = ['open', 'high', 'low', 'close']

    #scenario 1
    price =  np.array([1, 2, 3, 3, 4, 5, 6, 9, 8, 7, 9, 5, 4])
    df = build_test_df(price)
    env = Environment(df, features, meta_cols, actions=[0, 1, 2], min_obs=0,
                      add_features=0, skip_state=False, process_feats=False)
    action =  np.array([0, 2, 0, 1, 2, 2, 0, 1, 0, 1, 0, 0, 0])
    expected_rewards = np.array([0, 2, 0, 1, 2, 2, 0, -1, 0, 1, 0, 0, 0])
    rewards, states = gather_rewards_and_states(env, action)
    assert np.array_equal(np.array(rewards), np.array(expected_rewards).astype(np.float64))
    assert np.array_equal(
        np.array(states).flatten(),
        np.array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 12.])
    )

def test_env_skip():
    features = ['secs']
    meta_cols = ['open', 'high', 'low', 'close']
    price =  np.array([1, 2, 3, 3, 4, 5, 6, 9, 8, 7, 9, 5, 4])
    df = build_test_df(price)
    env = Environment(df, features, meta_cols, actions=[0, 1, 2], min_obs=0,
                      add_features=0, skip_state=True, process_feats=False)
    action =  np.array([0, 2, 1, 2, 1, 0, 1, 0, 0, 0,0,0,0,0])
    expected_rewards = np.array([0, 2, 1, 2, -1, 0, 1, 0, 0, 0])
    rewards, states = gather_rewards_and_states(env, action)
    assert np.array_equal(np.array(rewards), np.array(expected_rewards).astype(np.float64))
    assert np.array_equal(
        np.array(states).flatten(),
        np.array([ 1.,  4.,  5.,  7.,  8.,  9., 10., 11., 12., 12.])
    )

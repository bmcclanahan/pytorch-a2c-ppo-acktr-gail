import torch
import numpy as np

def select_action(model, state, hxs, masks):
    state = torch.from_numpy(state).float()
    state_value, action, action_log_probs, hxs = model.act(
        state.unsqueeze(0), hxs, masks, deterministic=True, return_dist=False
    )
    action = action.squeeze()
    action_log_prob = action_log_probs.squeeze()
    return action.item(), hxs

def normalize(obs, mean, var):
    return (obs - mean) / np.sqrt(var)

def validate_date_wrap(date, model, normalizer, env):
    masks = torch.ones(model.base._hidden_size).unsqueeze(0)
    hxs = torch.zeros(model.base._hidden_size).unsqueeze(0)
    return validate_date(date, model, normalizer, env, masks, hxs)

def validate_date(date, model, normalizer, env, masks, hxs, iter=1):
    running_reward = 0
    last_rewards = []
    entry_times = []
    trade_actions = []
    exit_times = []
    positions = []
    env.set_date(date)
    continuous = False
    if env.action_space.__class__.__name__ == "Box":
        continuous = True
    state = env.reset()
    ep_reward = 0
    done = False
    in_trade = False
    trade_end_tm = 0
    while not done:
        # select action from policy
        action, hxs = select_action(model, normalize(state, normalizer.mean,
                                    normalizer.var), hxs, masks)
        #print('action ', action)
        if (env.actions[int(action)] != 0) and not in_trade:
            in_trade = True
            entry_times.append(env.day_df.iloc[env.cursor].timestamp)
            positions.append(np.sign(env.actions[int(action)] ))

        # take the action
        state, reward, done, info = env.step(int(action) if not continuous else [action])

        if in_trade and not env.in_trade:
            ep_reward += reward
            trade_actions.append(action)
            cursor = min(env.cursor, env.day_df.shape[0] - 1)
            exit_times.append(env.day_df.iloc[cursor].timestamp)
            in_trade = False
        if done:
            break

    #print('Validation episode {} reward: {:.2f}'.format(
    #      iter, ep_reward))
    return ep_reward, entry_times, exit_times, positions

def validator(model, normalizer, env):
    val_dates = sorted(env.unique_dates)
    total_rewards = []
    masks = torch.ones(model.base._hidden_size).unsqueeze(0)
    hxs = torch.zeros(model.base._hidden_size).unsqueeze(0)
    for iter, date in enumerate(val_dates):
        ep_reward, _, _, _ = validate_date(date, model, normalizer, env,
                                           masks, hxs, iter=iter)
        total_rewards.append(ep_reward)
        hxs *= 0
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f'validation mean reward: {mean_reward} std: {std_reward} total_trades: {len(total_rewards)}')
    return mean_reward

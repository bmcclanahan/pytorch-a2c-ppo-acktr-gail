import copy
import glob
import os
import time
import datetime
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy, MLPBaseSingle
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
from gym.envs.registration import register
from futures_env.environment import EnvironmentContinuous,  EnvFullCont
from torch.utils.tensorboard import SummaryWriter

from validation import validation

register(
    id='FuturesEnv-v0',
    entry_point='futures_env.environment:EnvSkipState',
)

register(
    id='FuturesEnvFull-v0',
    entry_point='futures_env.environment:EnvFull',
)

register(
    id='FuturesEnvFull-v1',
    entry_point='futures_env.environment:EnvFullV2',
)

register(
    id='FuturesEnvCont-v0',
    entry_point='futures_env.environment:EnvFullCont',
)

register(
    id='FuturesEnvContTraining-v0',
    entry_point='futures_env.environment:EnvFullContTraining',
)

register(
    id='FuturesEnvTraining-v0',
    entry_point='futures_env.environment:EnvSkipStateTraining',
)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    writer_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    writer = SummaryWriter(f'{args.tensor_board_log_dir}-{writer_date}')

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    validate = False
    if args.validation_dataset is not None:
        val_df = pd.read_parquet(args.validation_dataset)
        val_env = gym.make(args.env_name, df=val_df, set_date=False)
        validate = True

    train_df = None
    if args.training_dataset is not None:
        train_df = pd.read_parquet(args.training_dataset)

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    normalizer = None
    if args.load_saved_model is not None:
        print(f'loading saved model {args.load_saved_model}')
        model_data = torch.load(args.load_saved_model)
        actor_critic = model_data[0]
        normalizer = model_data[1]

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False,
                         env_df=train_df, normalizer=normalizer)

    if args.load_saved_model is None:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base=MLPBaseSingle,
            base_kwargs={
                'recurrent': args.recurrent_policy,
                'hidden_size': args.hidden_size,
                'activation_type': args.activation_type
            }
        )
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=100)

    start = time.time()
    num_updates = args.num_updates
    total_episodes = 0
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps): # max number of steps in a day
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    total_episodes += 1

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps): # made change here
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            rw_mean = np.mean(episode_rewards)
            rw_median = np.median(episode_rewards)
            rw_min = np.min(episode_rewards)
            rw_max = np.max(episode_rewards)
            print(
                "Updates {}, Episodes {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_episodes,
                        len(episode_rewards), rw_mean,
                        rw_median, rw_min,
                        rw_max, dist_entropy, value_loss,
                        action_loss))
            writer.add_scalar('mean reward', rw_mean, j)
            writer.add_scalar('median reward', rw_mean, j)
            writer.add_scalar('max reward', rw_max, j)
            writer.add_scalar('min reward', rw_min, j)

        if (((j % args.validation_interval) == 0) or (j == num_updates - 1)) and validate:
            normalizer = getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            if normalizer is None:
                normalizer = lambda: None
                normalizer.mean = 1
                normalizer.var = 1
            avg_reward = validation.validator(actor_critic, normalizer, val_env)
            writer.add_scalar('validation mean reward', avg_reward, j)


        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

if __name__ == "__main__":
    main()

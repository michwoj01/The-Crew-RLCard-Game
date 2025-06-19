import os
import argparse
import numpy as np

import rlcard
import torch
from rlcard.agents import DQNAgent, NFSPAgent
from rlcard.utils import (
    get_device,
    set_seed,
    Logger,
    plot_curve,
)
from rlcard.envs.registration import register

from src.bridge.env import CrewEnv


def reorganize(trajectories, payoffs):
    num_players = len(trajectories)
    new_trajectories = [[] for _ in range(num_players)]
    for player in range(num_players):
        for i in range(0, len(trajectories[player]) - 2, 2):
            if i == len(trajectories[player]) - 3:
                done = True
            else:
                done = False
            reward = payoffs[player][i//2]
            transition = trajectories[player][i:i + 3].copy()
            transition.insert(2, reward)
            transition.append(done)

            new_trajectories[player].append(transition)
    return new_trajectories

def tournament(env, num):
    payoffs = [0 for _ in range(env.num_players)]
    counter = 0
    good_games = 0
    bad_games = 0
    while counter < num:
        _, _payoffs = env.run(is_training=False)
        if _payoffs[0][-1] > 0:
            good_games += 1
        else:
            bad_games += 1
        for i, _ in enumerate(payoffs):
            payoffs[i] += _payoffs[i][-1]
        counter += 1
    print(f"Good games: {good_games}, Bad games: {bad_games}")
    for i, _ in enumerate(payoffs):
        payoffs[i] /= counter
    return payoffs

def train(args):
    set_seed(args.seed)

    register(
        env_id='crew',
        entry_point='env:CrewEnv',
    )

    env: CrewEnv = rlcard.make('crew', config={
        'num_players': args.num_players,
        'no_tasks': args.no_tasks,
        'seed': args.seed,
        'fixed_tasks': args.fixed_tasks,
        'skip_signals': args.skip_signals,
    })
    device = get_device()

    if args.algorithm == 'dqn':
        if args.load_checkpoint_path:
            agent = DQNAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path, weights_only=False))
        else:
            agent = DQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every,
                replay_memory_size=50000,
                replay_memory_init_size = 2000,
                epsilon_decay_steps=500_000,
                epsilon_start=0.15,
                epsilon_end=0.01,
                batch_size=64,
                mlp_layers=[128, 128],
                learning_rate=1e-4,
            )
    else:
        if args.load_checkpoint_path:
            agent = NFSPAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path, weights_only=False))
        else:
            agent = NFSPAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every,
                batch_size=512,
                q_replay_memory_init_size=1000,
                q_replay_memory_size=50000,
                q_epsilon_decay_steps=300_000,
                q_mlp_layers=[128, 128],
                q_update_target_estimator_every=5000,
                q_epsilon_start=0.12,
                q_epsilon_end=0.02,
                q_batch_size=128,
                q_train_every=2,
                hidden_layers_sizes=[512, 512],
                reservoir_buffer_capacity=50000,
                rl_learning_rate=0.0005,
                sl_learning_rate=0.0001,
                min_buffer_size_to_learn=5000
            )
    agents = [agent for _ in range(env.num_players)]
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(1, args.num_episodes + 1):
            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)

            for player_id, trajectory in enumerate(trajectories):
                for ts in trajectory:
                    agents[player_id].feed(ts)

            if episode % args.evaluate_every == 0:
                performance = tournament(env, args.num_eval_games)
                logger.log_performance(episode, np.mean(performance))

    model_file = f'dqn_model_player.pth'
    agent.save_checkpoint(args.log_dir, model_file)
    plot_curve(logger.csv_path, logger.fig_path, args.algorithm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=100000)
    parser.add_argument('--num_eval_games', type=int, default=200)
    parser.add_argument('--evaluate_every', type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='experiments/')
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--algorithm", type=str, default="dqn")
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--no_tasks", type=int, default=2)
    parser.add_argument("--fixed_tasks", type=bool, default=True)
    parser.add_argument("--skip_signals", type=bool, default=False)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

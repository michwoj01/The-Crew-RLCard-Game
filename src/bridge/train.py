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
        'algorithm': args.algorithm
    })
    device = get_device()

    if args.algorithm == 'dqn':
        if args.load_checkpoint_path:
            agent = DQNAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path, weights_only=False))
        else:
            agent = DQNAgent(
                replay_memory_size=args.replay_memory_size,
                replay_memory_init_size = args.replay_memory_init_size,
                update_target_estimator_every=args.update_target_estimator_every,
                discount_factor=args.discount_factor,
                epsilon_start=args.epsilon_start,
                epsilon_end=args.epsilon_end,
                epsilon_decay_steps=args.epsilon_decay_steps,
                batch_size=args.batch_size,
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                train_every=args.train_every,
                mlp_layers=args.mlp_layers,
                learning_rate=args.learning_rate,
                device=device,
                save_path=args.save_path,
                save_every=args.save_every
            )
    else:
        if args.load_checkpoint_path:
            agent = NFSPAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path, weights_only=False))
        else:
            agent = NFSPAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                hidden_layers_sizes=args.hidden_layers_sizes,
                reservoir_buffer_capacity=args.reservoir_buffer_capacity,
                anticipatory_param=args.anticipatory_param,
                batch_size=args.nfsp_batch_size,
                train_every=args.nfsp_train_every,
                rl_learning_rate=args.rl_learning_rate,
                sl_learning_rate=args.sl_learning_rate,
                min_buffer_size_to_learn=args.min_buffer_size_to_learn,
                q_replay_memory_size=args.replay_memory_size,
                q_replay_memory_init_size=args.replay_memory_init_size,
                q_update_target_estimator_every=args.update_target_estimator_every,
                q_discount_factor=args.discount_factor,
                q_epsilon_start=args.epsilon_start,
                q_epsilon_end=args.epsilon_end,
                q_epsilon_decay_steps=args.epsilon_decay_steps,
                q_batch_size=args.batch_size,
                q_train_every=args.train_every,
                q_mlp_layers=args.mlp_layers,
                evaluate_with=args.evaluate_with,
                device=device,
                save_path=args.save_path,
                save_every=args.save_every
            )
    agents = [agent for _ in range(env.num_players)]
    env.set_agents(agents)

    # Start training
    with Logger(args.save_path) as logger:
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
    agent.save_checkpoint(args.save_path, model_file)
    plot_curve(logger.csv_path, logger.fig_path, args.algorithm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # training parameters
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--num_episodes', type=int, default=100_000)
    parser.add_argument('--num_eval_games', type=int, default=200)
    parser.add_argument('--evaluate_every', type=int, default=1000)
    parser.add_argument("--load_checkpoint_path", type=str, default="")

    # DQN agent parameters
    parser.add_argument('--replay_memory_size', type=int, default=50_000)
    parser.add_argument('--replay_memory_init_size', type=int, default=1000)
    parser.add_argument('--update_target_estimator_every', type=int, default=10_000)
    parser.add_argument('--discount_factor', type=int, default=0.99)
    parser.add_argument('--epsilon_start', type=float, default=0.15)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--epsilon_decay_steps', type=int, default=500_000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_every', type=int, default=1)
    parser.add_argument('--mlp_layers', nargs='+', type=int, default=[128, 128])
    parser.add_argument('--learning_rate', type=float, default=1e04)
    parser.add_argument('--save_path', type=str, default='experiments/')
    parser.add_argument("--save_every", type=int, default=10_000)

    # NFSP agent parameters
    parser.add_argument('--hidden_layers_sizes', nargs='+', type=int, default=[512, 512])
    parser.add_argument('--reservoir_buffer_capacity',type=int, default=50_000)
    parser.add_argument('--anticipatory_param',type=float, default=0.1)
    parser.add_argument('--nfsp_batch_size',type=int, default=256)
    parser.add_argument('--nfsp_train_every',type=int, default=1)
    parser.add_argument('--rl_learning_rate',type=float, default=0.1)
    parser.add_argument('--sl_learning_rate',type=float, default=0.005)
    parser.add_argument('--min_buffer_size_to_learn',type=int, default=100)
    parser.add_argument('--evaluate_with',type=str, default='average_policy')

    #environment parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--algorithm", type=str, default="dqn")
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--no_tasks", type=int, default=1)
    parser.add_argument("--fixed_tasks", type=bool, default=True)
    parser.add_argument("--skip_signals", type=bool, default=True)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

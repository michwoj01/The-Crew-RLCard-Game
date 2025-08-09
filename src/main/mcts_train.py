import argparse
import os
import time

import numpy as np

from src.main.env import CrewEnv
from src.rlcard.agents import DQNAgent
from src.rlcard.agents.mcts_agent import MCTSAgent
from src.rlcard.envs import register, make
from src.rlcard.utils import set_seed, Logger, get_device


def run_mcts(args):
    set_seed(args.seed)

    register(
        env_id='crew',
        entry_point='env:CrewEnv',
    )

    env: CrewEnv = make('crew', config={
        'num_players': args.num_players,
        'no_tasks': args.no_tasks,
        'seed': args.seed,
        'fixed_tasks': args.fixed_tasks,
        'skip_signals': args.skip_signals,
        'algorithm': args.algorithm
    })

    dqn_agent = DQNAgent(
        replay_memory_size=5_000,
        replay_memory_init_size=500,
        update_target_estimator_every=10_000,
        discount_factor=0.99,
        epsilon_start=0.2,
        epsilon_end=0.05,
        epsilon_decay_steps=100_000,
        batch_size=64,
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        train_every=2,
        mlp_layers=[128, 128],
        learning_rate=1e-4,
        device=get_device(),
        save_path='experiments_inside/',
        save_every=10_000
    )

    agents = [MCTSAgent(env, dqn_agent, n_simulations=args.n_simulations) for _ in range(env.num_players)]
    env.set_agents(agents)

    won_games = 0

    with Logger(args.save_path) as _logger:
        for episode in range(1, args.num_episodes + 1):
            start_time = time.time()
            _, payoffs = env.run(is_training=False)
            if payoffs[0] == 1:
                won_games += 1
            print(payoffs, time.time() - start_time)

    print('won games: ', won_games)
    print('total games: ', args.num_episodes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--num_eval_games', type=int, default=100)
    parser.add_argument('--evaluate_every', type=int, default=10)
    parser.add_argument("--n_simulations", type=int, default=1000)
    parser.add_argument('--save_path', type=str, default='experiments/')

    # environment parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--algorithm", type=str, default="mcts")
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--no_tasks", type=int, default=4)
    parser.add_argument("--fixed_tasks", type=bool, default=False)
    parser.add_argument("--skip_signals", type=bool, default=True)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    run_mcts(args)

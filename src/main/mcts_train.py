import argparse
import os
import time

import numpy as np

from src.main.env import CrewEnv
from src.rlcard.agents.mcts_agent import MCTSAgent
from src.rlcard.envs import register, make
from src.rlcard.utils import set_seed, Logger


def run_mcts(args):
    set_seed(args.seed)

    register(
        env_id='crew',
        entry_point='env:CrewEnv',
    )

    env: CrewEnv = make('crew', config={
        'num_players': args.num_players,
        'no_tasks': args.no_tasks,
        'skip_signals': args.skip_signals,
        'algorithm': args.algorithm
    })

    agents = [MCTSAgent(env, n_simulations=args.n_simulations) for _ in range(env.num_players)]
    env.set_agents(agents)

    with Logger(args.save_path) as logger:
        for episode in range(1, args.num_episodes + 1):
            start_time = time.time()
            _, payoffs = env.run(is_training=False)
            print(payoffs, time.time() - start_time)
            logger.log_performance(episode, np.sum(payoffs[0]) / 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--num_eval_games', type=int, default=20)
    parser.add_argument('--evaluate_every', type=int, default=10)
    parser.add_argument("--n_simulations", type=int, default=1000)
    parser.add_argument('--save_path', type=str, default='experiments/')

    # environment parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--algorithm", type=str, default="mcts")
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--no_tasks", type=int, default=4)
    parser.add_argument("--skip_signals", type=bool, default=False)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    run_mcts(args)

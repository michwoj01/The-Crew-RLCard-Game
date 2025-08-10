import argparse
import os

from src.rlcard.agents import MCTSAgent
from src.rlcard.envs import register, make, CrewEnv
from src.rlcard.utils import set_seed, Logger, tournament_with_eval_hands


def run_mcts(args):
    set_seed(args.seed)

    register(
        env_id='crew',
        entry_point='src.rlcard.envs.env:CrewEnv',
    )

    env: CrewEnv = make('crew', config={
        'num_players': args.num_players,
        'no_tasks': args.no_tasks,
        'skip_signals': args.skip_signals,
        'algorithm': args.algorithm
    })

    with Logger(args.save_path, 'mcts2') as logger:
        for sims in range(100, 1001, 100):
            agents = [MCTSAgent(env, n_simulations=sims) for _ in range(env.num_players)]
            env.set_agents(agents)
            performance = tournament_with_eval_hands(env)
            logger.log_performance(sims, performance)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--num_eval_games', type=int, default=20)
    parser.add_argument('--evaluate_every', type=int, default=10)
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

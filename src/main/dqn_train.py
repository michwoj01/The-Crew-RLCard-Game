import argparse
import time

import numpy as np
import torch

from src.main.env import CrewEnv
from src.rlcard.agents import DQNAgent, NFSPAgent
from src.rlcard.agents.mcts_agent import MCTSAgent
from src.rlcard.envs import register, make
from src.rlcard.utils import set_seed, Logger, plot_curve, reorganize, tournament, get_device


def train(args):
    device = get_device()
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

    if args.algorithm == 'dqn':
        if args.load_checkpoint_path:
            agent = DQNAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path, weights_only=False))
        else:
            agent = DQNAgent(
                replay_memory_size=args.replay_memory_size,
                replay_memory_init_size=args.replay_memory_init_size,
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
    # FOR 1s DGN AND 3 MCTS
    # for _ in range(env.num_players - 1):
    #     agents.append(MCTSAgent(env, n_simulations=args.n_simulations))
    env.set_agents(agents)

    # Start training
    with Logger(args.save_path) as logger:
        start_time = time.time()
        for episode in range(1, args.num_episodes + 1):
            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)

            for ts in trajectories[0]:
                agents[0].feed(ts)
            if episode % args.evaluate_every == 0:
                performance = tournament(env, args.num_eval_games)
                logger.log_performance(episode, np.mean(performance))
                elapsed = time.time() - start_time
                print(f"Epoki {episode - 999}–{episode} ukończone w {elapsed:.2f} s")
                start_time = time.time()

    model_file = f'dqn_model_player.pth'
    agent.save_checkpoint(args.save_path, model_file)
    plot_curve(logger.csv_path, logger.fig_path, args.algorithm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument('--num_episodes', type=int, default=300_000)  # 1000
    parser.add_argument('--num_eval_games', type=int, default=100)  # 5
    parser.add_argument('--evaluate_every', type=int, default=1000)  # 20
    parser.add_argument("--load_checkpoint_path", type=str, default="")

    # DQN agent parameters
    parser.add_argument('--replay_memory_size', type=int, default=5_000)
    parser.add_argument('--replay_memory_init_size', type=int, default=500)
    parser.add_argument('--update_target_estimator_every', type=int, default=10_000)
    parser.add_argument('--discount_factor', type=int, default=0.99)
    parser.add_argument('--epsilon_start', type=float, default=0.2)
    parser.add_argument('--epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon_decay_steps', type=int, default=100_000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_every', type=int, default=2)
    parser.add_argument('--mlp_layers', nargs='+', type=int, default=[128, 128])
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_path', type=str, default='experiments/')
    parser.add_argument("--save_every", type=int, default=10_000)

    # NFSP agent parameters
    parser.add_argument('--hidden_layers_sizes', nargs='+', type=int, default=[512, 512])
    parser.add_argument('--reservoir_buffer_capacity', type=int, default=50_000)
    parser.add_argument('--anticipatory_param', type=float, default=0.1)
    parser.add_argument('--nfsp_batch_size', type=int, default=256)
    parser.add_argument('--nfsp_train_every', type=int, default=1)
    parser.add_argument('--rl_learning_rate', type=float, default=0.1)
    parser.add_argument('--sl_learning_rate', type=float, default=0.005)
    parser.add_argument('--min_buffer_size_to_learn', type=int, default=100)
    parser.add_argument('--evaluate_with', type=str, default='average_policy')

    # environment parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--algorithm", type=str, default="dqn")
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--no_tasks", type=int, default=2)
    parser.add_argument("--fixed_tasks", type=bool, default=False)
    parser.add_argument("--skip_signals", type=bool, default=False)
    parser.add_argument("--n_simulations", type=int, default=100)

    args = parser.parse_args()

    train(args)

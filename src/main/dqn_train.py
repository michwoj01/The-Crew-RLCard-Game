import argparse
import time

import numpy as np

from src.main.env import CrewEnv
from src.rlcard.agents import DQNAgent
from src.rlcard.agents.mcts_agent import MCTSAgent
from src.rlcard.envs import register, make
from src.rlcard.utils import set_seed, Logger, plot_curve, reorganize, get_device, \
    tournament_with_eval_hands


def get_dqn_parameters(num_dqn_agents):
    if num_dqn_agents == 1:  # 1 DQN vs 3 MCTS
        return {
            'learning_rate': 1e-3,  # Higher LR - environment is stationary
            'update_target_estimator_every': 1000,
            'epsilon_start': 0.9,
            'epsilon_end': 0.05,
            'epsilon_decay_steps': 20000,  # Faster decay
            'replay_memory_size': 20000,  # Smaller buffer needed
            'replay_memory_init_size': 1000,
            'train_every': 1,  # Train every step
            'batch_size': 64,
            'num_episodes': 15000,  # Fewer episodes needed
            'evaluate_every': 500,
            'num_eval_games': 10,
            'mlp_layers': [256, 128],  # Simpler network
            'discount_factor': 0.99
        }

    elif num_dqn_agents == 2:  # 2 DQN vs 2 MCTS/DQN
        return {
            'learning_rate': 7e-4,  # Moderate LR
            'update_target_estimator_every': 1500,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay_steps': 40000,  # Moderate decay
            'replay_memory_size': 50000,  # Medium buffer
            'replay_memory_init_size': 1500,
            'train_every': 2,  # Train every 2 steps
            'batch_size': 64,
            'num_episodes': 25000,
            'evaluate_every': 750,
            'num_eval_games': 10,
            'mlp_layers': [256, 256],  # Medium network
            'discount_factor': 0.97
        }

    elif num_dqn_agents == 3:  # 3 DQN vs 1 MCTS
        return {
            'learning_rate': 6e-4,  # Lower LR
            'update_target_estimator_every': 1750,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay_steps': 60000,  # Slower decay
            'replay_memory_size': 75000,  # Larger buffer
            'replay_memory_init_size': 2000,
            'train_every': 3,  # Train every 3 steps
            'batch_size': 64,
            'num_episodes': 35000,
            'evaluate_every': 1000,
            'num_eval_games': 10,
            'mlp_layers': [256, 256, 128],  # Deeper network
            'discount_factor': 0.96
        }

    elif num_dqn_agents == 4:  # 4 DQN vs 0 MCTS
        return {
            'learning_rate': 5e-4,  # Lowest LR - most non-stationary
            'update_target_estimator_every': 2000,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay_steps': 100000,  # Slowest decay
            'replay_memory_size': 100000,  # Largest buffer
            'replay_memory_init_size': 2500,
            'train_every': 4,  # Train every 4 steps
            'batch_size': 64,
            'num_episodes': 50000,  # Most episodes
            'evaluate_every': 1000,
            'num_eval_games': 10,
            'mlp_layers': [256, 256, 128],  # Deepest network
            'discount_factor': 0.95
        }

    else:
        raise ValueError(f"Unsupported configuration: {num_dqn_agents} DQN agents")


def create_agents_with_progressive_params(args, env, device):
    num_dqn = args.num_dqn_agents
    num_mcts = args.num_players - num_dqn

    params = get_dqn_parameters(num_dqn)

    agents = []

    # Create DQN agents with progressive parameters
    agent = DQNAgent(
        replay_memory_size=params['replay_memory_size'],
        replay_memory_init_size=params['replay_memory_init_size'],
        update_target_estimator_every=params['update_target_estimator_every'],
        discount_factor=params['discount_factor'],
        epsilon_start=params['epsilon_start'],
        epsilon_end=params['epsilon_end'],
        epsilon_decay_steps=params['epsilon_decay_steps'],
        batch_size=params['batch_size'],
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        train_every=params['train_every'],
        mlp_layers=params['mlp_layers'],
        learning_rate=params['learning_rate'],
        device=device,
        save_path=args.save_path,
        save_every=args.save_every
    )
    for i in range(num_dqn):
        agents.append(agent)

    # Add MCTS agents
    for i in range(num_mcts):
        agents.append(MCTSAgent(env, n_simulations=args.n_simulations))

    return agents, params


# You could also implement a progressive schedule
def get_eval_frequency(episode, params):
    eval_every = params['evaluate_every']
    if episode < 10000:
        return 2 * eval_every  # Less frequent early on when learning is unstable
    elif episode < 30000:
        return eval_every  # Standard frequency during main learning
    else:
        return eval_every / 2  # More frequent near the end to catch convergence


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
        'skip_signals': args.skip_signals,
        'algorithm': args.algorithm,
        'is_clone': False
    })

    agents, params = create_agents_with_progressive_params(args, env, device)
    env.set_agents(agents)
    # Start training
    with Logger(args.save_path) as logger:
        start_time = time.time()
        for episode in range(1, params['num_episodes'] + 1):
            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)

            # Train all agents
            for player_id in range(env.num_players):
                for ts in trajectories[player_id]:
                    agents[player_id].feed(ts)

            if episode % get_eval_frequency(episode, params) == 0:
                performance = tournament_with_eval_hands(env)
                logger.log_performance(episode, np.mean(performance))
                elapsed = time.time() - start_time
                # print(f"Episode {episode} completed in {elapsed:.2f} s")
                start_time = time.time()

    # Save all agents
    for player_id, agent in enumerate(agents):
        model_file = f'dqn_model_player_{player_id}.pth'
        agent.save_checkpoint(args.save_path, model_file)

    plot_curve(logger.csv_path, logger.fig_path, args.algorithm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument('--save_path', type=str, default='experiments/')
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--algorithm", type=str, default="dqn")
    parser.add_argument("--num_players", type=int, default=4)
    parser.add_argument("--num_dqn_agents", type=int, default=4)
    parser.add_argument("--no_tasks", type=int, default=1)
    parser.add_argument("--skip_signals", type=bool, default=False)
    parser.add_argument("--n_simulations", type=int, default=100)

    args = parser.parse_args()

    train(args)

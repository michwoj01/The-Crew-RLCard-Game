import os
import argparse

import numpy as np
import torch

import rlcard
from rlcard.agents import DQNAgent, NFSPAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
from rlcard.envs.registration import register


def train(args):

    set_seed(args.seed)

    register(
        env_id='crew',
        entry_point='env:CrewEnv',
    )

    env = rlcard.make('crew', config={'seed': args.seed})

    # Check whether gpu is available
    device = get_device()

    agents = []

    # Initialize the agent and use random agents as opponents
    for index in range(env.num_players):
        # # Determine the checkpoint path for this particular agent
        # filename = f'model_{index}.pth'
        # checkpoint_path = os.path.join(args.load_checkpoint_path, filename)

        # # Load the agent from the checkpoint if it exists
        # if os.path.exists(checkpoint_path):
        #     print(f"Loading checkpoint for agent {index} from {checkpoint_path}")
        #     if args.algorithm == 'dqn':
        #         agent = DQNAgent.from_checkpoint(checkpoint=torch.load(checkpoint_path))
        #     elif args.algorithm == 'nfsp':
        #         agent = NFSPAgent.from_checkpoint(checkpoint=torch.load(checkpoint_path))
        # else:
        if args.algorithm == 'dqn':
            agent = DQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[64, 64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )
        elif args.algorithm == 'nfsp':
            agent = NFSPAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                hidden_layers_sizes=[64, 64],
                q_mlp_layers=[64, 64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )

        agents.append(agent)

    print(agents)
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):

            if args.algorithm == 'nfsp':
                for agent in agents:
                    agent.sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for idx, trajectory in enumerate(trajectories):
                print(len(trajectory), idx)
                for ts in trajectory:
                    agents[idx].feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    episode,
                    np.average(
                        tournament(
                            env,
                            args.num_eval_games,
                        )
                    )
                )

    # Get the paths
    csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    for index, agent in enumerate(agents):
        filename = f'model_{index:02d}.pth'
        save_path = os.path.join(args.log_dir, filename)
        torch.save(agent, save_path)
        print('Model saved in', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument(
        '--algorithm',
        type=str,
        default='dqn',
        choices=[
            'dqn',
            'nfsp',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=3000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/',
    )

    parser.add_argument(
        "--load_checkpoint_path",
        type=str,
        default="",
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=-1)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

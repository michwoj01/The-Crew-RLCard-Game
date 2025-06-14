import os
import argparse
import numpy as np

import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)
from rlcard.envs.registration import register

from shared_dqn_agent import SharedDQNAgent


def train(args):
    set_seed(args.seed)

    register(
        env_id='crew',
        entry_point='env:CrewEnv',
    )

    env = rlcard.make('crew', config={'seed': args.seed})
    device = get_device()

    # Create shared agent
    shared_agent = DQNAgent(
                    num_actions=env.num_actions,
                    state_shape=env.state_shape[0],
                    mlp_layers=[64, 64],
                    device=device,
                    save_path=args.log_dir,
                    save_every=args.save_every
                )

    agents = [shared_agent for _ in range(env.num_players)]
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(1, args.num_episodes + 1):
            print(f'Episode: {episode}')
            trajectories, payoffs = env.run(is_training=True)

            # Reorganize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for player_id, trajectory in enumerate(trajectories):
                for ts in trajectory:
                    shared_agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                performance = tournament(env, args.num_eval_games)
                logger.log_performance(episode, np.mean(performance))

    # Save model
    model_path = os.path.join(args.log_dir, 'shared_dqn_model.pth')
    shared_agent.save(model_path)
    print('Model saved at:', model_path)

    plot_curve(logger.csv_path, logger.fig_path, args.algorithm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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

import os
import argparse
import numpy as np

import rlcard
from rlcard.agents import DQNAgent
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
    while counter < num:
        _, _payoffs = env.run(is_training=False)
        if isinstance(_payoffs, list):
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i][-1]
            counter += 1
        else:
            for i, _ in enumerate(payoffs):
                payoffs[i] += _payoffs[i]
            counter += 1
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
        'seed': args.seed
    })
    device = get_device()

    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[64, 64],
        device=device,
        save_path=args.log_dir,
        save_every=args.save_every
    )

    agents = [agent for _ in range(env.num_players)]
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(1, args.num_episodes + 1):
            print(f'Episode: {episode}')
            trajectories, payoffs = env.run(is_training=True)
            trajectories = reorganize(trajectories, payoffs)

            for player_id, trajectory in enumerate(trajectories):
                for ts in trajectory:
                    agents[player_id].feed(ts)

            if episode % args.evaluate_every == 0:
                performance = tournament(env, args.num_eval_games)
                logger.log_performance(episode, np.mean(performance))

    for i, agent in enumerate(agents):
        model_file = f'dqn_model_player_{i}.pth'
        agent.save_checkpoint(args.log_dir, model_file)
        print(f'Model gracza {i} zapisany w: {model_file}')

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

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

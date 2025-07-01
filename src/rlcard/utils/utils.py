import random

import numpy as np
import torch

from src.rlcard.utils.base import Card


def set_seed(seed):
    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def get_device():
    import torch
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print("--> Running on the MPS (Apple Silicon) GPU")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    return device


def init_standard_deck():
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    return res


def init_54_deck():
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    res.append(Card('BJ', ''))
    res.append(Card('RJ', ''))
    return res


def rank2int(rank):
    if rank == '':
        return -1
    elif rank.isdigit():
        if 2 <= int(rank) <= 10:
            return int(rank)
        else:
            return None
    elif rank == 'A':
        return 14
    elif rank == 'T':
        return 10
    elif rank == 'J':
        return 11
    elif rank == 'Q':
        return 12
    elif rank == 'K':
        return 13
    return None


def elegent_form(card):
    suits = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣', 's': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    rank = '10' if card[1] == 'T' else card[1]

    return suits[card[0]] + rank


def print_card(cards):
    if cards is None:
        cards = [None]
    if isinstance(cards, str):
        cards = [cards]

    lines = [[] for _ in range(9)]

    for card in cards:
        if card is None:
            lines[0].append('┌─────────┐')
            lines[1].append('│░░░░░░░░░│')
            lines[2].append('│░░░░░░░░░│')
            lines[3].append('│░░░░░░░░░│')
            lines[4].append('│░░░░░░░░░│')
            lines[5].append('│░░░░░░░░░│')
            lines[6].append('│░░░░░░░░░│')
            lines[7].append('│░░░░░░░░░│')
            lines[8].append('└─────────┘')
        else:
            if isinstance(card, Card):
                elegent_card = elegent_form(card.suit + card.rank)
            else:
                elegent_card = elegent_form(card)
            suit = elegent_card[0]
            rank = elegent_card[1]
            if len(elegent_card) == 3:
                space = elegent_card[2]
            else:
                space = ' '

            lines[0].append('┌─────────┐')
            lines[1].append('│{}{}       │'.format(rank, space))
            lines[2].append('│         │')
            lines[3].append('│         │')
            lines[4].append('│    {}    │'.format(suit))
            lines[5].append('│         │')
            lines[6].append('│         │')
            lines[7].append('│       {}{}│'.format(space, rank))
            lines[8].append('└─────────┘')

    for line in lines:
        print('   '.join(line))


def reorganize(trajectories, payoffs):
    num_players = len(trajectories)
    new_trajectories = [[] for _ in range(num_players)]
    for player in range(num_players):
        for i in range(0, len(trajectories[player]) - 2, 2):
            if i == len(trajectories[player]) - 3:
                done = True
            else:
                done = False
            reward = payoffs[player][i // 2]
            transition = trajectories[player][i:i + 3].copy()
            transition.insert(2, reward)
            transition.append(done)

            new_trajectories[player].append(transition)
    return new_trajectories


def remove_illegal(action_probs, legal_actions):
    probs = np.zeros(action_probs.shape[0])
    probs[legal_actions] = action_probs[legal_actions]
    if np.sum(probs) == 0:
        probs[legal_actions] = 1 / len(legal_actions)
    else:
        probs /= sum(probs)
    return probs


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

def plot_curve(csv_path, save_path, algorithm):
    import os
    import csv
    import matplotlib.pyplot as plt
    with open(csv_path) as csvfile:
        reader = csv.DictReader(csvfile)
        xs = []
        ys = []
        for row in reader:
            xs.append(int(row['episode']))
            ys.append(float(row['reward']))
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label=algorithm)
        ax.set(xlabel='episode', ylabel='reward')
        ax.legend()
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)

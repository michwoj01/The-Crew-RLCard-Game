import numpy as np
from action_event import ActionEvent


class HumanAgent(object):

    def __init__(self, num_actions):
        self.use_raw = False
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        _print_state(state)
        action = int(input('>> You choose action (integer): '))
        while not isinstance(action, int) or action < 0 or action >= len(state['legal_actions']):
            print('Action illegal...')
            action = int(input('>> Re-choose action (integer): '))
        return state['raw_legal_actions'][action]

    def eval_step(self, state):
        return self.step(state), {}


def _print_state(state):
    raw_obs = state['obs']
    raw_legal_actions = state['raw_legal_actions']

    offset = 0

    hand = [i for i in range(40) if raw_obs[offset + i] == 1]
    print(f"\nYour hand ({len(hand)}): ", end="")
    print(", ".join([f"{ActionEvent.from_action_id(card_id)}" for card_id in hand]))
    offset += 40

    print("\nCurrent trick:")
    for pid in range(4):
        trick_rep = raw_obs[offset + pid * 14: offset + (pid + 1) * 14]
        rank_part = trick_rep[:9]
        suit_part = trick_rep[9:14]
        player_cards = []
        for rank_idx, v in enumerate(rank_part):
            if v == 1:
                for suit_idx, s in enumerate(suit_part):
                    if s == 1:
                        player_cards.append((rank_idx, suit_idx))
        if player_cards:
            print(f"  Player {pid}: ", end="")
            print(", ".join([str(ActionEvent.from_action_id(rank + 9 * suit)) for rank, suit in player_cards]))
    offset += 4 * 14

    # Tasks (4 x 13)
    print("\nTasks:")
    for pid in range(4):
        task_rep = raw_obs[offset + pid * 13: offset + (pid + 1) * 13]
        rank_part = task_rep[:9]
        suit_part = task_rep[9:13]
        player_tasks = []
        for rank_idx, v in enumerate(rank_part):
            if v == 1:
                for suit_idx, s in enumerate(suit_part):
                    if s == 1:
                        player_tasks.append((rank_idx, suit_idx))
        if player_tasks:
            print(f"  Player {pid}: ", end="")
            print(", ".join([str(ActionEvent.from_action_id(rank + 9 * suit)) for rank, suit in player_tasks]))
    offset += 4 * 13

    # hidden_cards = [i for i in range(40) if raw_obs[offset + i] == 1]
    # print("\nHidden cards:")
    # for card_id in hidden_cards:
    #     print(f"  {ActionEvent.from_action_id(card_id)}")
    offset += 40
    offset += 4

    if offset + 4 * 17 <= len(raw_obs):
        print("\nSignals:")
        for pid in range(4):
            sig_rep = raw_obs[offset + pid * 17: offset + (pid + 1) * 17]
            rank_idx = np.argmax(sig_rep[:9]) if np.any(sig_rep[:9]) else None
            suit_idx = np.argmax(sig_rep[9:13]) if np.any(sig_rep[9:13]) else None
            signal_type = np.argmax(sig_rep[14:17]) if np.any(sig_rep[14:17]) else None
            if rank_idx is not None and suit_idx is not None and signal_type is not None:
                card = ActionEvent.from_action_id(76 + signal_type * 40 + suit_idx * 9 + rank_idx)
                print(f"  Player {pid} signals {card} with signal type {signal_type}")

    print("\nLegal actions:")
    for idx, action_id in enumerate(raw_legal_actions):
        print(f"  {idx}: Play {ActionEvent.from_action_id(action_id)}")

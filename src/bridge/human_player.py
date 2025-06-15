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
        while action < 0 or action >= len(state['legal_actions']):
            print('Action illegal...')
            action = int(input('>> Re-choose action (integer): '))
        return state['raw_legal_actions'][action]

    def eval_step(self, state):
        return self.step(state), {}


def _print_state(state):
    raw_obs = state['raw_obs']
    raw_legal_actions = state['raw_legal_actions']

    current_player_id = np.argmax(raw_obs[-4:])
    hand_start = current_player_id * 40
    hand = [i for i in range(40) if raw_obs[hand_start + i] == 1]

    print(f"\nTwoje karty ({len(hand)}):")
    for idx, card_id in enumerate(hand):
        print(f"  {idx}: {ActionEvent.from_action_id(card_id)}")

    print("\nAktualna lewa:")
    trick_pile_offset = 4 * 40
    for pid in range(4):
        player_trick = [i for i in range(40) if raw_obs[trick_pile_offset + pid * 40 + i] == 1]
        if player_trick:
            for card_id in player_trick:
                print(f"  Gracz {pid}: {ActionEvent.from_action_id(card_id)}")

    print("\nAktualne taski:")
    tasks_offset = 8 * 40
    for pid in range(4):
        player_tasks = [i for i in range(36) if raw_obs[tasks_offset + pid * 36 + i] == 1]
        if player_tasks:
            for card_id in player_tasks:
                print(f"  Gracz {pid}: {ActionEvent.from_action_id(card_id)}")

    print("\nSygnały graczy:")
    signals_offset = tasks_offset + 4 * 36
    for pid in range(4):
        player_signals = [i for i in range(120) if raw_obs[signals_offset + pid * 120 + i] == 1]
        if player_signals:
            for sig in player_signals:
                card_id = sig % 40
                signal_type = sig // 40
                print(f"  Gracz {pid}: {ActionEvent.from_action_id(card_id)}, typ sygnału: {signal_type}")

    print("\nMożliwe akcje:")
    for idx, action_id in enumerate(raw_legal_actions):
        print(f"  {idx}: Zagraj {ActionEvent.from_action_id(action_id)} (action_id={action_id})")

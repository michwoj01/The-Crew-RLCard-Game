import numpy as np

from src.rlcard.envs.action_event import ActionEvent


class HumanAgent(object):

    def __init__(self):
        self.use_raw = False

    @staticmethod
    def step(state) -> int:
        _print_state(state)
        action = int(input('>> You choose action (integer): '))
        while not isinstance(action, int) or action < 0 or action >= len(state['legal_actions']):
            print('Action illegal...')
            action = int(input('>> Re-choose action (integer): '))
        return state['legal_actions'][action]

    def eval_step(self, state):
        return self.step(state), {}


from colorama import Fore, Style, init

init(autoreset=True)

def _print_state(state):
    raw_obs = state['obs']
    raw_legal_actions = state['legal_actions']
    offset = 0

    color_map = {
        'B': Fore.BLUE,
        'Y': Fore.YELLOW,
        'P': Fore.MAGENTA,
        'G': Fore.GREEN,
        'R': Fore.BLACK,
        '': Fore.WHITE,
        's': Fore.WHITE,
    }

    def fmt_card(card):
        card_str = str(ActionEvent.from_action_id(card))  # e.g., Y5, R2, or "Play skip signal"
        suit = card_str[0] if card_str and card_str[0] in color_map else ''
        return f"{color_map[suit]}{card_str}{Style.RESET_ALL}"

    def fmt_signal_type(sig_char):
        return {'L': 'Low', 'H': 'High', 'O': 'Only'}.get(sig_char, sig_char)

    def sort_cards(card_ids):
        # Sort: suit (B/Y/P/R), then rank
        def key_fn(card_id):
            card = str(ActionEvent.from_action_id(card_id))
            suit_order = {'B': 0, 'Y': 1, 'P': 2, 'R': 3}
            suit = card[0]
            rank = int(card[1]) if len(card) == 2 else int(card[1:])
            return (suit_order.get(suit, 9), rank)

        return sorted(card_ids, key=key_fn)

    def print_card_grid(title, card_ids):
        print(f"\n{title} ({len(card_ids)}):")
        cards = [fmt_card(cid) for cid in sort_cards(card_ids)]
        for i, card in enumerate(cards):
            print(f"  {card:<6}", end="\n" if (i + 1) % 8 == 0 else "")
        if len(cards) % 8 != 0:
            print()

    # Ręka
    hand = [i for i in range(40) if raw_obs[offset + i] == 1]
    print_card_grid("Your hand", hand)
    offset += 40

    # Trick
    print("\nCurrent trick:")
    for pid in range(4):
        trick_rep = raw_obs[offset + pid * 14: offset + (pid + 1) * 14]
        rank_part = trick_rep[:9]
        suit_part = trick_rep[9:14]
        for rank_idx, r in enumerate(rank_part):
            if r == 1:
                for suit_idx, s in enumerate(suit_part):
                    if s == 1:
                        card_id = rank_idx + 9 * suit_idx
                        print(f"  Player {pid}: {fmt_card(card_id)}")
    offset += 4 * 14

    # Tasks
    print("\nTasks:")
    for pid in range(4):
        task_rep = raw_obs[offset + pid * 14: offset + (pid + 1) * 14]
        rank_part = task_rep[:9]
        suit_part = task_rep[9:13]
        for rank_idx, r in enumerate(rank_part):
            if r == 1:
                for suit_idx, s in enumerate(suit_part):
                    if s == 1:
                        card_id = rank_idx + 9 * suit_idx
                        print(f"  Player {pid}: {fmt_card(card_id)}")
    offset += 4 * 14 + 40 + 4

    # Signals
    if offset + 4 * 17 <= len(raw_obs):
        print("\nSignals:")
        for pid in range(4):
            sig_rep = raw_obs[offset + pid * 17: offset + (pid + 1) * 17]
            rank_idx = np.argmax(sig_rep[:9]) if np.any(sig_rep[:9]) else None
            suit_idx = np.argmax(sig_rep[9:13]) if np.any(sig_rep[9:13]) else None
            sig_type_idx = np.argmax(sig_rep[14:17]) if np.any(sig_rep[14:17]) else None
            if rank_idx is not None and suit_idx is not None and sig_type_idx is not None:
                card_id = 76 + sig_type_idx * 40 + suit_idx * 9 + rank_idx
                card = str(ActionEvent.from_action_id(card_id))
                suit = card[0]
                rank = card[1:-1]
                sig_char = card[-1]
                print(
                    f"  Player {pid} signal: {color_map[suit]}{suit}{rank}{Style.RESET_ALL} - {fmt_signal_type(sig_char)}")

    # Legal actions
    print("\nLegal actions:")
    for idx, action_id in enumerate(raw_legal_actions):
        card = str(ActionEvent.from_action_id(action_id))
        if "signal" in card.lower():
            suit = card[0]
            sig_char = card[-1]
            print(f"  {idx}: Signal {color_map[suit]}{card[:-1]}{Style.RESET_ALL} - {fmt_signal_type(sig_char)}")
        else:
            print(f"  {idx}: Play {fmt_card(action_id)}")

import random

import numpy as np

from card import CrewCard, CrewTask
from player import CrewPlayer


class Dealer:
    # 10 hardcoded evaluation handouts for consistent evaluation
    EVAL_HANDOUTS = [
        {
            'hands': [
                # Player 0
                ['Y4', 'G5', 'B7', 'P9', 'G2', 'B3', 'P1', 'Y8', 'G6', 'B4'],
                # Player 1
                ['Y2', 'G7', 'B9', 'P3', 'G4', 'B6', 'Y9', 'P7', 'G8', 'B1'],
                # Player 2
                ['Y6', 'G1', 'B5', 'P8', 'G9', 'B2', 'Y7', 'P4', 'G3', 'B8'],
                # Player 3
                ['Y1', 'G0', 'B0', 'P2', 'Y3', 'P5', 'Y5', 'P6', 'P0', 'Y0']
            ],
            'tasks': ['G2', 'B3', 'P1', 'Y8']
        },
        {
            'hands': [
                # Player 0
                ['Y7', 'G3', 'B1', 'P5', 'G8', 'B9', 'P2', 'Y1', 'G4', 'B6'],
                # Player 1
                ['Y9', 'G6', 'B4', 'P8', 'G1', 'B7', 'Y3', 'P9', 'G5', 'B2'],
                # Player 2
                ['Y2', 'G9', 'B8', 'P1', 'G7', 'B3', 'Y6', 'P4', 'G0', 'B5'],
                # Player 3
                ['Y4', 'G2', 'B0', 'P7', 'Y8', 'P3', 'Y5', 'P6', 'P0', 'Y0']
            ],
            'tasks': ['G3', 'B1', 'P5', 'Y7']
        },
        {
            'hands': [
                # Player 0
                ['Y3', 'G7', 'B2', 'P6', 'G4', 'B8', 'P3', 'Y9', 'G1', 'B5'],
                # Player 1
                ['Y6', 'G2', 'B9', 'P1', 'G8', 'B4', 'Y2', 'P7', 'G5', 'B3'],
                # Player 2
                ['Y8', 'G9', 'B1', 'P9', 'G3', 'B6', 'Y7', 'P2', 'G6', 'B7'],
                # Player 3
                ['Y1', 'G0', 'B0', 'P5', 'Y4', 'P8', 'Y5', 'P4', 'P0', 'Y0']
            ],
            'tasks': ['G7', 'B2', 'P6', 'Y3']
        },
        {
            'hands': [
                # Player 0
                ['Y8', 'G1', 'B6', 'P4', 'G9', 'B2', 'P8', 'Y3', 'G5', 'B9'],
                # Player 1
                ['Y2', 'G8', 'B3', 'P7', 'G2', 'B5', 'Y6', 'P1', 'G4', 'B7'],
                # Player 2
                ['Y9', 'G3', 'B4', 'P2', 'G6', 'B8', 'Y1', 'P9', 'G7', 'B1'],
                # Player 3
                ['Y7', 'G0', 'B0', 'P3', 'Y4', 'P6', 'Y5', 'P5', 'P0', 'Y0']
            ],
            'tasks': ['G1', 'B6', 'P4', 'Y8']
        },
        {
            'hands': [
                # Player 0
                ['Y5', 'G4', 'B7', 'P9', 'G1', 'B4', 'P5', 'Y8', 'G8', 'B1'],
                # Player 1
                ['Y3', 'G7', 'B2', 'P6', 'G9', 'B8', 'Y1', 'P2', 'G2', 'B6'],
                # Player 2
                ['Y6', 'G5', 'B9', 'P3', 'G3', 'B5', 'Y9', 'P8', 'G6', 'B3'],
                # Player 3
                ['Y2', 'G0', 'B0', 'P1', 'Y7', 'P7', 'Y4', 'P4', 'P0', 'Y0']
            ],
            'tasks': ['G4', 'B7', 'P9', 'Y5']
        },
        {
            'hands': [
                # Player 0
                ['Y1', 'G6', 'B8', 'P2', 'G5', 'B1', 'P7', 'Y4', 'G3', 'B5'],
                # Player 1
                ['Y7', 'G1', 'B3', 'P8', 'G4', 'B9', 'Y6', 'P3', 'G8', 'B2'],
                # Player 2
                ['Y3', 'G9', 'B6', 'P5', 'G2', 'B4', 'Y8', 'P1', 'G7', 'B7'],
                # Player 3
                ['Y9', 'G0', 'B0', 'P9', 'Y2', 'P6', 'Y5', 'P4', 'P0', 'Y0']
            ],
            'tasks': ['G6', 'B8', 'P2', 'Y1']
        },
        {
            'hands': [
                # Player 0
                ['Y9', 'G2', 'B4', 'P7', 'G6', 'B9', 'P1', 'Y2', 'G4', 'B3'],
                # Player 1
                ['Y4', 'G8', 'B1', 'P4', 'G3', 'B6', 'Y7', 'P9', 'G1', 'B8'],
                # Player 2
                ['Y6', 'G7', 'B2', 'P8', 'G9', 'B7', 'Y3', 'P3', 'G5', 'B5'],
                # Player 3
                ['Y8', 'G0', 'B0', 'P2', 'Y1', 'P5', 'Y5', 'P6', 'P0', 'Y0']
            ],
            'tasks': ['G2', 'B4', 'P7', 'Y9']
        },
        {
            'hands': [
                # Player 0
                ['Y7', 'G9', 'B5', 'P3', 'G2', 'B8', 'P6', 'Y5', 'G6', 'B1'],
                # Player 1
                ['Y1', 'G4', 'B9', 'P7', 'G8', 'B2', 'Y9', 'P1', 'G3', 'B4'],
                # Player 2
                ['Y8', 'G1', 'B3', 'P9', 'G7', 'B6', 'Y2', 'P4', 'G5', 'B7'],
                # Player 3
                ['Y3', 'G0', 'B0', 'P8', 'Y6', 'P2', 'Y4', 'P5', 'P0', 'Y0']
            ],
            'tasks': ['G9', 'B5', 'P3', 'Y7']
        },
        {
            'hands': [
                # Player 0
                ['Y2', 'G5', 'B7', 'P1', 'G3', 'B3', 'P9', 'Y6', 'G9', 'B6'],
                # Player 1
                ['Y8', 'G2', 'B4', 'P5', 'G7', 'B9', 'Y3', 'P4', 'G6', 'B1'],
                # Player 2
                ['Y4', 'G8', 'B2', 'P6', 'G1', 'B5', 'Y9', 'P7', 'G4', 'B8'],
                # Player 3
                ['Y7', 'G0', 'B0', 'P3', 'Y1', 'P8', 'Y5', 'P2', 'P0', 'Y0']
            ],
            'tasks': ['G5', 'B7', 'P1', 'Y2']
        },
        {
            'hands': [
                # Player 0
                ['Y6', 'G8', 'B1', 'P8', 'G4', 'B7', 'P2', 'Y9', 'G2', 'B4'],
                # Player 1
                ['Y5', 'G6', 'B8', 'P4', 'G1', 'B3', 'Y8', 'P6', 'G9', 'B5'],
                # Player 2
                ['Y1', 'G3', 'B9', 'P1', 'G7', 'B2', 'Y4', 'P9', 'G5', 'B6'],
                # Player 3
                ['Y3', 'G0', 'B0', 'P7', 'Y2', 'P3', 'Y7', 'P5', 'P0', 'Y0']
            ],
            'tasks': ['G8', 'B1', 'P8', 'Y6']
        }
    ]

    def __init__(self, np_random, no_tasks: int = 4, eval_mode: bool = False, eval_hand_id: int = 0):
        self.eval_mode = eval_mode
        self.eval_hand_id = eval_hand_id
        if eval_mode:
            if eval_hand_id >= len(self.EVAL_HANDOUTS):
                raise ValueError(
                    f"eval_hand_id {eval_hand_id} is out of range. Available: 0-{len(self.EVAL_HANDOUTS) - 1}")

            eval_data = self.EVAL_HANDOUTS[eval_hand_id]

            self.eval_hands = []
            for hand_strings in eval_data['hands']:
                hand = []
                for card_str in hand_strings:
                    suit = card_str[0]
                    rank = int(card_str[1])
                    hand.append(CrewCard(suit=suit, rank=rank))
                self.eval_hands.append(hand)

            self.tasks = []
            for task_str in eval_data['tasks'][:no_tasks]:
                suit = task_str[0]
                rank = int(task_str[1])
                self.tasks.append(CrewCard(suit=suit, rank=rank))

            self.shuffled_deck = CrewCard.get_deck()
            self.stock_pile = self.shuffled_deck.copy()

        else:
            if np_random:
                self.np_random = np_random if np_random is not None else np.random.default_rng()
                self.shuffled_deck: list[CrewCard] = CrewCard.get_deck()
                self.np_random.shuffle(self.shuffled_deck)
                self.stock_pile: list[CrewCard] = self.shuffled_deck.copy()
                self.np_random.shuffle(self.shuffled_deck)
                self.tasks: list[CrewCard] = random.sample([card for card in self.shuffled_deck
                                                            if card.suit != CrewCard.trump_suit], no_tasks)

    def deal_cards(self, player: CrewPlayer, num: int):
        if self.eval_mode:
            if len(self.eval_hands) <= player.player_id:
                raise ValueError(f"No predefined hand for player {player.player_id}")

            hand = self.eval_hands[player.player_id]
            if len(hand) < num:
                raise ValueError(
                    f"Predefined hand for player {player.player_id} has only {len(hand)} cards, but {num} requested")
            for i in range(num):
                player.hand.append(hand[i])
        else:
            for _ in range(num):
                player.hand.append(self.stock_pile.pop())

    def assign_task(self, player_id: int, card: CrewCard) -> CrewTask:
        task = CrewTask(card=card, owner=player_id)
        self.tasks.remove(card)
        return task

    def clone(self):
        cloned = Dealer(np_random=None)
        cloned.shuffled_deck = [card for card in self.shuffled_deck]
        cloned.stock_pile = [card for card in self.stock_pile]
        cloned.tasks = [card for card in self.tasks]
        if self.eval_mode:
            cloned.eval_hands = [[card for card in hand] for hand in self.eval_hands]
        return cloned

    @classmethod
    def get_num_eval_handouts(cls):
        """Return the number of available evaluation handouts"""
        return len(cls.EVAL_HANDOUTS)

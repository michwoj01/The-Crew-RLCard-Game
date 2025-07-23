import random

import numpy as np

from card import CrewCard, CrewTask
from player import CrewPlayer


class Dealer:

    def __init__(self, np_random, no_tasks: int = 4, fixed_tasks: bool = False):
        if np_random:
            self.np_random = np_random if np_random is not None else np.random.default_rng()
            self.shuffled_deck: list[CrewCard] = CrewCard.get_deck()
            self.np_random.shuffle(self.shuffled_deck)
            self.stock_pile: list[CrewCard] = self.shuffled_deck.copy()
            if fixed_tasks:
                fixed_tasks = [CrewCard(suit='Y', rank=8), CrewCard(suit='G', rank=2), CrewCard(suit='B', rank=3),
                               CrewCard(suit='P', rank=1)]
                self.tasks = fixed_tasks[:no_tasks]
            else:
                self.np_random.shuffle(self.shuffled_deck)
                self.tasks_pile: list[CrewCard] = [card for card in self.shuffled_deck if
                                                   card.suit != CrewCard.trump_suit]
                self.tasks: list[CrewCard] = random.sample(self.tasks_pile, no_tasks)

    def deal_cards(self, player: CrewPlayer, num: int):
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
        return cloned

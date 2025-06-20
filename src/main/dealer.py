import random

from card import CrewCard, CrewTask
from player import CrewPlayer


class Dealer:

    def __init__(self, no_tasks: int, np_random, fixed_tasks: bool = False):
        self.np_random = np_random
        self.shuffled_deck: list[CrewCard] = CrewCard.get_deck()
        self.np_random.shuffle(self.shuffled_deck)
        self.stock_pile: list[CrewCard] = self.shuffled_deck.copy()
        self.tasks_pile = [CrewCard(suit='Y', rank=8), CrewCard(suit='G', rank=2)]
        if fixed_tasks:
            self.tasks = self.tasks_pile[:no_tasks]
        else:
            self.np_random.shuffle(self.shuffled_deck)
            self.tasks: list[CrewCard] = random.sample(self.tasks_pile, no_tasks)

    def deal_cards(self, player: CrewPlayer, num: int):
        for _ in range(num):
            player.hand.append(self.stock_pile.pop())

    def assign_task(self, player_id: int, card: CrewCard) -> CrewTask:
        task = CrewTask(card=card, owner=player_id)
        self.tasks.remove(card)
        return task

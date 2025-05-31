import random

from card import CrewCard
from player import CrewPlayer


class Dealer:

    def __init__(self, np_random):
        self.np_random = np_random
        self.shuffled_deck: list[CrewCard] = CrewCard.get_deck()
        self.np_random.shuffle(self.shuffled_deck)
        self.stock_pile: list[CrewCard] = self.shuffled_deck.copy()
        self.np_random.shuffle(self.shuffled_deck)
        self.tasks_pile: list[CrewCard] = [card for card in self.shuffled_deck if card.suit != 'R']
        self.tasks: list[CrewCard] = []

    def deal_cards(self, player: CrewPlayer, num: int):
        for _ in range(num):
            player.hand.append(self.stock_pile.pop())

    def prepare_tasks(self):
        self.tasks = random.sample(self.tasks_pile, 4)

    def assign_task(self, player: CrewPlayer, task: CrewCard):
        player.assign_task(task)
        self.tasks.remove(task)

from typing import List
from card import CrewCard
from player import CrewPlayer


class Dealer:

    def __init__(self, np_random):
        self.np_random = np_random
        self.shuffled_deck: List[CrewCard] = CrewCard.get_deck()
        self.np_random.shuffle(self.shuffled_deck)
        self.stock_pile: List[CrewCard] = self.shuffled_deck.copy()
        self.np_random.shuffle(self.shuffled_deck)
        self.tasks_pile: List[CrewCard] = [card for card in self.shuffled_deck if card.suit != 'R'][:4]

    def deal_cards(self, player: CrewPlayer, num: int):
        for _ in range(num):
            player.hand.append(self.stock_pile.pop())

    def deal_tasks(self, player: CrewPlayer):
        player.tasks_assigned.append(self.tasks_pile.pop())

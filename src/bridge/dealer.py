from typing import List
from card import CrewCard
from players import CrewPlayer


class Dealer:

    def __init__(self, np_random):
        self.np_random = np_random
        self.shuffled_deck: List[CrewCard] = CrewCard.get_deck()
        self.np_random.shuffle(self.shuffled_deck)
        self.stock_pile: List[CrewCard] = self.shuffled_deck.copy()

    def deal_cards(self, player: CrewPlayer, num: int):
        for _ in range(num):
            player.hand.append(self.stock_pile.pop())

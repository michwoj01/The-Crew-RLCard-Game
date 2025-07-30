from enum import Enum

from base import Card


class SignalType(Enum):
    LOWEST = 0
    HIGHEST = 1
    ONLY = 2

    def __str__(self):
        return self.name[0]


class CrewCard(Card):
    suits = ['B', 'G', 'Y', 'P', 'R']
    full_suits = ['Blue', 'Green', 'Yellow', 'Pink', 'Rocket']
    trump_suit = 'R'
    ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __init__(self, suit: str, rank: int):
        super().__init__(suit=suit, rank=rank)
        self.suit_index = CrewCard.suits.index(suit)
        self.rank_index = CrewCard.ranks.index(rank)
        self.card_id = 9 * self.suit_index + self.rank_index

    @staticmethod
    def card(card_id: int):
        return _deck[card_id]

    @staticmethod
    def get_deck():
        return _deck.copy()

    def __str__(self):
        return f'{self.suit}{self.rank}'

    def __repr__(self):
        return f'{self.suit}{self.rank}'

    def full_suit_name(self):
        return f'{CrewCard.full_suits[self.suit_index]}-{self.rank}'


_deck = [CrewCard(suit=suit, rank=rank) for suit in CrewCard.suits[:4] for rank in CrewCard.ranks] + \
        [CrewCard(CrewCard.trump_suit, rank) for rank in range(1, 5)]

class CrewTask:
    def __init__(self, card: CrewCard, owner: int):
        self.card: CrewCard = card
        self.owner: int = owner
        self.taken: bool = False
        self.taker: int = -1

    def complete(self, taker: int) -> bool:
        if self.taken:
            raise Exception(f'Task {self.card} already completed')
        self.taker = taker
        self.taken = True
        if self.owner == taker:
            return True
        else:
            return False

    def clone(self) -> 'CrewTask':
        new_task = CrewTask(self.card, self.owner)
        new_task.taken = self.taken
        new_task.taker = self.taker
        return new_task

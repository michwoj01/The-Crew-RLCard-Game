from enum import Enum


class Signal(Enum):
    LOWEST = 1
    HIGHEST = 2
    ONLY = 3

    # implement printing Signal
    def __str__(self):
        return self.name


class CrewCard:
    def __init__(self, suit: str, rank: int):
        self.suit: str = suit
        self.rank: int = rank
        self.is_rocket: bool = suit == 'R'

    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank

    def __str__(self):
        return f'{self.suit}{self.rank}'


Communicate = tuple[CrewCard, Signal]

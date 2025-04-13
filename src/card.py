from enum import Enum


class Signal(Enum):
    LOWEST = 1
    HIGHEST = 2
    ONLY = 3

    def fromLetter(letter: str):
        if letter == 'L':
            return Signal.LOWEST
        elif letter == 'H':
            return Signal.HIGHEST
        elif letter == 'O':
            return Signal.ONLY
        else:
            raise ValueError(f"Invalid signal letter: {letter}")

    # implement printing Signal
    def __str__(self):
        return self.name[0]


class CrewCard:
    def __init__(self, suit: str, rank: int):
        self.suit: str = suit
        self.rank: int = rank
        self.is_rocket: bool = suit == 'R'

    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank

    def __str__(self):
        return f'{self.suit}{self.rank}'
    
    def to_tuple(self):
        return (self.suit, self.rank)


Communicate = tuple[CrewCard, Signal]

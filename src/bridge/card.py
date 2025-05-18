from rlcard.games.base import Card


class CrewCard(Card):
    suits = ['B', 'G', 'Y', 'P', 'R']
    ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def __init__(self, suit: str, rank: int):
        super().__init__(suit=suit, rank=rank)
        suit_index = CrewCard.suits.index(suit)
        rank_index = CrewCard.ranks.index(rank)
        self.card_id = 9 * suit_index + rank_index

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


_deck = [CrewCard(suit=suit, rank=rank) for suit in CrewCard.suits[:4] for rank in CrewCard.ranks] + \
        [CrewCard('R', rank) for rank in range(1, 5)]

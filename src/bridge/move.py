from action_event import *
from card import CrewCard
from player import CrewPlayer


class CrewMove(object):  # Interface
    pass


class PlayerMove(CrewMove):  # Interface

    def __init__(self, player: CrewPlayer, action: ActionEvent):
        super().__init__()
        self.player = player
        self.action = action


class DealHandMove(CrewMove):

    def __init__(self, dealer: CrewPlayer, shuffled_deck: [CrewCard]):
        super().__init__()
        self.dealer = dealer
        self.shuffled_deck = shuffled_deck

    def __str__(self):
        shuffled_deck_text = " ".join([str(card) for card in self.shuffled_deck])
        return f'{self.dealer} deal shuffled_deck=[{shuffled_deck_text}]'


class DealTaskHandMove(CrewMove):

    def __init__(self, dealer: CrewPlayer, tasks: [CrewCard]):
        super().__init__()
        self.dealer = dealer
        self.tasks = tasks

    def __str__(self):
        shuffled_deck_text = " ".join([str(card) for card in self.tasks])
        return f'{self.dealer} deal tasks=[{shuffled_deck_text}]'


class PlayCardMove(PlayerMove):

    def __init__(self, player: CrewPlayer, action: PlayCardAction):
        super().__init__(player=player, action=action)
        self.action = action

    @property
    def card(self):
        return self.action.card

    def __str__(self):
        return f'{self.player} plays {self.action}'

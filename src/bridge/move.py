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

    def __init__(self, dealer: CrewPlayer, shuffled_deck: list[CrewCard]):
        super().__init__()
        self.dealer = dealer
        self.shuffled_deck = shuffled_deck

    def __str__(self):
        shuffled_deck_text = " ".join([str(card)
                                      for card in self.shuffled_deck])
        return f'{self.dealer} deal shuffled_deck=[{shuffled_deck_text}]'


class PlayCardMove(PlayerMove):

    def __init__(self, player: CrewPlayer, action: PlayCardAction):
        super().__init__(player=player, action=action)

    @property
    def card(self):
        return self.action.card

    def __str__(self):
        return f'{self.player} plays {self.action}'


class SignalMove(PlayerMove):

    def __init__(self, player: CrewPlayer, action: SignalAction):
        super().__init__(player=player, action=action)

    @property
    def signal(self):
        return (self.action.card, self.action.signal_type)

    def __str__(self):
        return f'{self.player} signals {self.action}'


class SkipMove(PlayerMove):

    def __init__(self, player: CrewPlayer, action: SkipSignalAction):
        super().__init__(player=player, action=action)

    def __str__(self):
        return f'{self.player} skips signal'


class ChooseTaskMove(PlayerMove):

    def __init__(self, player: CrewPlayer, action: ChooseTaskAction):
        super().__init__(player=player, action=action)
        self.action = action

    @property
    def task(self):
        return self.action.task

    def __str__(self):
        return f'{self.player} chooses {self.action}'

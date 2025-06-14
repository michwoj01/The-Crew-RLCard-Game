from action_event import *
from card import CrewCard

class CrewMove(object):  # Interface
    pass


class PlayerMove(CrewMove):  # Interface

    def __init__(self, player_id: int, action: ActionEvent):
        super().__init__()
        self.player_id = player_id
        self.action = action


class DealHandMove(CrewMove):

    def __init__(self, shuffled_deck: list[CrewCard]):
        super().__init__()
        self.shuffled_deck = shuffled_deck

    def __str__(self):
        shuffled_deck_text = " ".join([str(card)
                                      for card in self.shuffled_deck])
        return f'shuffled_deck=[{shuffled_deck_text}]'


class PlayCardMove(PlayerMove):

    def __init__(self, player_id: int, action: PlayCardAction):
        super().__init__(player_id=player_id, action=action)

    @property
    def card(self):
        return self.action.card

    def __str__(self):
        return f'{self.player_id} plays {self.action}'


class SignalMove(PlayerMove):

    def __init__(self, player_id: int, action: SignalAction):
        super().__init__(player_id=player_id, action=action)

    @property
    def signal(self):
        return (self.action.card, self.action.signal_type)

    def __str__(self):
        return f'{self.player_id} signals {self.action}'


class SkipMove(PlayerMove):

    def __init__(self, player_id: int, action: SkipSignalAction):
        super().__init__(player_id=player_id, action=action)

    def __str__(self):
        return f'{self.player_id} skips signal'


class ChooseTaskMove(PlayerMove):

    def __init__(self, player_id: int, action: ChooseTaskAction):
        super().__init__(player_id=player_id, action=action)
        self.action = action

    @property
    def task(self):
        return self.action.card

    def __str__(self):
        return f'{self.player_id} chooses {self.action}'

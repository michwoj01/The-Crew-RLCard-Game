from action_event import *


class CrewMove(object):  # Interface
    pass


class PlayerMove(CrewMove):  # Interface

    def __init__(self, player_id: int, action: ActionEvent):
        super().__init__()
        self.player_id = player_id
        self.action = action


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

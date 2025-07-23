from card import CrewCard, SignalType


class CrewPlayer:

    def __init__(self, player_id: int):
        if player_id < 0 or player_id > 3:
            raise Exception(f'CrewPlayer has invalid player_id: {player_id}')
        self.player_id: int = player_id
        self.hand: list[CrewCard] = []
        self.signal: tuple[CrewCard, SignalType] = None

    def remove_card_from_hand(self, card: CrewCard):
        self.hand.remove(card)

    def can_signal(self):
        return self.signal is None and len(self.hand) > 0

    def signal_card(self, card: CrewCard, signal_type: SignalType):
        if self.signal is None:
            self.signal = (card, signal_type)

    def __str__(self):
        return str(self.player_id)

    def clone(self) -> 'CrewPlayer':
        new_player = CrewPlayer(player_id=self.player_id)
        new_player.hand = [card for card in self.hand]
        new_player.signal = self.signal if self.signal is None else (self.signal[0], self.signal[1])
        return new_player

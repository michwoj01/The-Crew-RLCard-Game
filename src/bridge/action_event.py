from card import CrewCard, Signal


class ActionEvent(object):

    first_play_card_action_id = 0
    first_play_signal_action_id = 40
    num_signals = 3
    max_signal_uses = 1

    def __str__(self):
        return f"{self.card} ({self.signal})"

    def __repr__(self):
        return self.__str__()

    def __init__(self, action_id: int, card: CrewCard, signal: Signal | None):
        self.action_id = action_id
        self.card = card
        self.signal = signal

    @staticmethod
    def get_num_actions():
        return 160

    @staticmethod
    def is_play_card_action(action_id: int):
        return ActionEvent.first_play_card_action_id <= action_id < ActionEvent.first_play_signal_action_id

    @staticmethod
    def is_signal_card_action(action_id: int):
        return action_id >= ActionEvent.first_play_signal_action_id

    @staticmethod
    def get_action_id_for_play_card(card: CrewCard):
        action_id = ActionEvent.first_play_card_action_id + card.card_id
        return ActionEvent(action_id, card, None)

    @staticmethod
    def get_action_id_for_signal_card(card: CrewCard, signal: Signal):
        action_id = ActionEvent.first_play_signal_action_id + card.card_id * ActionEvent.num_signals + signal.value
        return ActionEvent(action_id, card, signal)

    @staticmethod
    def decode_action_id(action_id: int):
        if ActionEvent.is_play_card_action(action_id):
            card = action_id - ActionEvent.first_play_card_action_id
            return card, None
        elif ActionEvent.is_signal_card_action(action_id):
            relative_id = action_id - ActionEvent.first_play_signal_action_id
            card = relative_id // ActionEvent.num_signals
            signal_id = relative_id % ActionEvent.num_signals
            return card, signal_id
        else:
            raise ValueError("Invalid action_id")

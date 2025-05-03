# Action_ids:
#       0 to 39 -> play_card_action_id
#       40 to 159 -> signal_action_id (40 cards * 3 signals: only, lowest, highest)

class ActionEvent(object):  # Interface

    first_play_card_action_id = 0
    first_play_signal_action_id = 40
    num_signals = 3  # only, lowest, highest
    max_signal_uses = 1

    def __str__(self):
        signals = ["only", "lowest", "highest"]
        return f"{self.card} ({signals[self.signal_id]})"

    def __repr__(self):
        return self.__str__()


    def __init__(self, action_id, card=None, signal_id=None):
        self.action_id = action_id
        self.card = card
        self.signal_id = signal_id

    @staticmethod
    def is_play_card_action(action_id):
        return ActionEvent.first_play_card_action_id <= action_id < ActionEvent.first_play_signal_action_id

    @staticmethod
    def is_signal_card_action(action_id):
        return action_id >= ActionEvent.first_play_signal_action_id

    @staticmethod
    def get_action_id_for_play_card(card):
        return ActionEvent.first_play_card_action_id + card

    @staticmethod
    def get_action_id_for_signal_card(card, signal_id):
        return ActionEvent.first_play_signal_action_id + card * ActionEvent.num_signals + signal_id

    @staticmethod
    def decode_action_id(action_id):
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
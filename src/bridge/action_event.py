from card import CrewCard


class ActionEvent(object):
    first_play_card_action_id = 0
    first_play_signal_action_id = 40

    def __init__(self, action_id: int):
        self.action_id = action_id

    def __eq__(self, other):
        result = False
        if isinstance(other, ActionEvent):
            result = self.action_id == other.action_id
        return result

    @staticmethod
    def from_action_id(action_id: int):
        card = CrewCard.card(card_id=action_id)
        return PlayCardAction(card=card)

    @staticmethod
    def get_num_actions():
        return 40


class PlayCardAction(ActionEvent):

    def __init__(self, card: CrewCard):
        play_card_action_id = card.card_id
        super().__init__(action_id=play_card_action_id)
        self.card: CrewCard = card

    def __str__(self):
        return f"{self.card}"

    def __repr__(self):
        return f"{self.card}"

from card import CrewCard, SignalType


class ActionEvent(object):
    first_play_card_action_id = 0
    first_choose_task_action_id = 40
    first_signal_action_id = 76
    last_signal_action_id = 195
    skip_signal_action_id = 196

    def __init__(self, action_id: int):
        self.action_id = action_id

    def __eq__(self, other):
        result = False
        if isinstance(other, ActionEvent):
            result = self.action_id == other.action_id
        return result

    @staticmethod
    def from_action_id(action_id: int):
        if action_id < ActionEvent.first_choose_task_action_id:
            card = CrewCard.card(card_id=action_id)
            return PlayCardAction(card=card)
        elif action_id < ActionEvent.first_signal_action_id:
            card = CrewCard.card(card_id=action_id -
                                 ActionEvent.first_choose_task_action_id)
            return ChooseTaskAction(card=card)
        elif action_id <= ActionEvent.last_signal_action_id:
            signal_type_id = (
                action_id - ActionEvent.first_signal_action_id) // 40
            card_id = (action_id - ActionEvent.first_signal_action_id) % 40
            return SignalAction(CrewCard.card(card_id=card_id), SignalType(signal_type_id))
        elif action_id == ActionEvent.skip_signal_action_id:
            return SkipSignalAction()
        else:
            raise ValueError(f"Invalid action_id: {action_id}")

    @staticmethod
    def get_num_actions():
        return 197


class PlayCardAction(ActionEvent):

    def __init__(self, card: CrewCard):
        play_card_action_id = card.card_id
        super().__init__(action_id=play_card_action_id)
        self.card: CrewCard = card

    def __str__(self):
        return f"card - {self.card}"

    def __repr__(self):
        return f"card - {self.card}"


class ChooseTaskAction(ActionEvent):

    def __init__(self, card: CrewCard):
        choose_task_action_id = ActionEvent.first_choose_task_action_id + card.card_id
        super().__init__(action_id=choose_task_action_id)
        self.card: CrewCard = card

    def __str__(self):
        return f"task - {self.card}"

    def __repr__(self):
        return f"task - {self.card}"


class SignalAction(ActionEvent):

    def __init__(self, card: CrewCard, signal_type: SignalType):
        card_id = card.card_id
        signal_type_id = signal_type.value
        signal_id = ActionEvent.first_signal_action_id + signal_type_id * 40 + card_id
        super().__init__(action_id=signal_id)
        self.card: CrewCard = card
        self.signal_type: SignalType = signal_type

    def __str__(self):
        return f"signal - {self.card} - {self.signal_type}"

    def __repr__(self):
        return f"signal - {self.card} - {self.signal_type}"


class SkipSignalAction(ActionEvent):

    def __init__(self):
        super().__init__(action_id=ActionEvent.skip_signal_action_id)

    def __str__(self):
        return "skip signal"

    def __repr__(self):
        return "skip signal"

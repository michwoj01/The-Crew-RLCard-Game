from card import CrewCard


class ActionEvent(object):
    first_play_card_action_id = 0
    first_choose_task_action_id = 40
    first_signal_action_id = 76

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
            task = CrewCard.card(card_id=action_id - ActionEvent.first_choose_task_action_id)
            return ChooseTaskAction(task=task)
        else:
            # TODO
            return None

    @staticmethod
    def get_num_actions():
        return 76


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

    def __init__(self, task: CrewCard):
        choose_task_action_id = ActionEvent.first_choose_task_action_id + task.card_id
        super().__init__(action_id=choose_task_action_id)
        self.task: CrewCard = task

    def __str__(self):
        return f"task - {self.task}"

    def __repr__(self):
        return f"task - {self.task}"

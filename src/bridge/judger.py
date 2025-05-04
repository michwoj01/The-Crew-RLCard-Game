from typing import List
from game import CrewGame
from action_event import ActionEvent


class Judger:

    def __init__(self, game: 'CrewGame'):
        self.game: CrewGame = game

    def get_legal_actions(self) -> List[ActionEvent]:
        legal_actions: List[ActionEvent] = []
        raise NotImplementedError("Judger.get_legal_actions() is not implemented")

import numpy as np
from utils.card import CrewCard, Communicate, Signal
from rlcard.agents import RandomAgent


class CrewPlayer(RandomAgent):

    def __init__(self, num_actions, player_id: int):
        super().__init__(num_actions)
        self.player_id: int = player_id
        self.has_communicated: bool = False
        self.hand: list[CrewCard] = []
        self.missions: list[CrewCard] = []

    def choose_task(self, tasks: list[CrewCard]) -> CrewCard:
        return np.random.choice(tasks)

    def complete_mission(self, card: CrewCard):
        if card in self.missions:
            self.missions.remove(card)

    def __str__(self):
        return f'Player {self.player_id}'

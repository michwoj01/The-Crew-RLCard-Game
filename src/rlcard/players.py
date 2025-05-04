import numpy as np
from utils.card import CrewCard
from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents.random_agent import RandomAgent


class CrewPlayer:
    def __init__(self, player_id: int):
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


class RandomCrewPlayer(RandomAgent, CrewPlayer):

    def __init__(self, player_id: int):
        RandomAgent.__init__(self, num_actions=40)
        CrewPlayer.__init__(self, player_id)


class NFSPCrewPlayer(NFSPAgent, CrewPlayer):

    def __init__(self, player_id: int):
        NFSPAgent.__init__(self, num_actions=40, state_shape=[40], hidden_layers_sizes=[
                           512, 512], q_mlp_layers=[512, 512])
        CrewPlayer.__init__(self, player_id)

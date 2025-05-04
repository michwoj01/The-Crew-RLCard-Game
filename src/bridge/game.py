import numpy as np
from judger import Judger
from action_event import ActionEvent
from round import Round


class CrewGame:
    def __init__(self,):
        self.np_random = np.random.RandomState()
        self.judger: Judger = Judger(game=self)
        self.actions: list[ActionEvent] = []
        self.num_players: int = 4

    def init_game(self):
        ''' Initialize all characters in the game and start round 1
        '''
        self.actions: list[ActionEvent] = []
        self.round = Round(
            num_players=self.num_players,
            np_random=self.np_random)
        for player_id in range(4):
            player = self.round.players[player_id]
            self.round.dealer.deal_cards(player=player, num=13)
        current_player_id = self.round.current_player_id
        state = self.get_state(player_id=current_player_id)
        return state, current_player_id

    def step(self, action: ActionEvent):
        self.round.play_card(action=action)
        self.actions.append(action)
        next_player_id = self.round.current_player_id
        next_state = self.get_state(player_id=next_player_id)
        return next_state, next_player_id

    def get_num_players(self) -> int:
        return self.num_players

    def get_player_id(self)-> int:
        return self.round.current_player_id

    def is_over(self) -> bool:
        return self.round.is_over()

    def get_state(self, player_id: int):
        state = {}
        state['player_id'] = player_id
        state['current_player_id'] = self.round.current_player_id
        state['hand'] = self.round.players[player_id].hand
        return state

import numpy as np
from judger import Judger
from action_event import ActionEvent, ChooseTaskAction, SignalAction, SkipSignalAction
from round import Round


class CrewGame:
    def __init__(self):
        self.allow_step_back = False
        self.np_random = np.random.RandomState(seed=42)
        self.judger: Judger = Judger(game=self)
        self.actions: list[ActionEvent] = []  # must reset in init_game
        self.round: Round = None  # must reset in init_game
        self.num_players: int = 4

    def init_game(self):
        self.actions: list[ActionEvent] = []
        self.round = Round(
            num_players=self.num_players,
            np_random=self.np_random)
        for player_id in range(4):
            player = self.round.players[player_id]
            self.round.dealer.deal_cards(player=player, num=10)
        self.round.dealer.prepare_tasks()
        current_player_id = self.round.current_player_id
        state = self.get_state(player_id=current_player_id)
        return state, current_player_id

    def step(self, action: ActionEvent):
        if isinstance(action, ChooseTaskAction):
            self.round.choose_task(action=action)
        elif isinstance(action, SignalAction) or isinstance(action, SkipSignalAction):
            self.round.signal(action=action)
        else:
            self.round.play_card(action=action)
        self.actions.append(action)
        next_player_id = self.round.current_player_id
        next_state = self.get_state(player_id=next_player_id)
        return next_state, next_player_id

    def get_num_players(self) -> int:
        return self.num_players

    @staticmethod
    def get_num_actions() -> int:
        return ActionEvent.get_num_actions()

    def get_player_id(self) -> int:
        return self.round.current_player_id

    def is_over(self) -> bool:
        return self.round.is_over()

    # stub implementation, we use DefaultCrewStateExtractor
    def get_state(self, player_id: int):
        state = {'player_id': player_id, 'current_player_id': self.round.current_player_id,
                 'hand': self.round.players[player_id].hand}
        return state

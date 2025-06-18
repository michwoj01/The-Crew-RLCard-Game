import numpy as np
from action_event import ActionEvent, ChooseTaskAction, SignalAction, SkipSignalAction
from round import Round


class CrewGame:
    def __init__(self, num_players: int = 4, no_tasks: int = 4, seed: int = 42,
                 fixed_tasks: bool = False, skip_signals: bool = False):
        self.allow_step_back = False
        self.round: Round = None
        self.np_random = np.random.RandomState(seed=seed)
        self.num_players: int = num_players
        self.no_tasks: int = no_tasks
        self.fixed_tasks: bool = fixed_tasks
        self.skip_signals: bool = skip_signals

    def init_game(self):
        self.round = Round(
            num_players=self.num_players,
            no_tasks=self.no_tasks,
            np_random=self.np_random,
            fixed_tasks=self.fixed_tasks)
        self.round.init_round()
        current_player_id = self.get_player_id()
        return {}, current_player_id

    def step(self, action: ActionEvent):
        if isinstance(action, ChooseTaskAction):
            self.round.choose_task(action=action)
        elif isinstance(action, SignalAction) or isinstance(action, SkipSignalAction):
            self.round.signal(action=action)
        else:
            self.round.play_card(action=action)
        next_player_id = self.get_player_id()
        return {}, next_player_id

    def get_num_players(self) -> int:
        return self.num_players

    def get_num_actions(self) -> int:
        return ActionEvent.get_num_actions(self.skip_signals)

    def get_player_id(self) -> int:
        return self.round.get_current_player_id()

    def is_over(self) -> bool:
        return self.round.is_over()

    # stub implementation
    def get_state(self, player_id: int):
        return {}

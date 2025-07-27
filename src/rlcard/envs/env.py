import numpy as np

from src.rlcard.envs.action_event import ActionEvent, ChooseTaskAction, SignalAction, SkipSignalAction
from src.rlcard.envs.game import Game
from src.rlcard.envs.judger import Judger


class CrewEnv:
    def __init__(self, config):
        self.agents = None
        self.game: Game = None
        self.name = 'crew'
        self.eval_mode = config.get('eval_mode', False)
        self.eval_hand_id = config.get('eval_hand_id', 0)
        self.skip_signals = config.get('skip_signals', False)
        self.algorithm = config.get('algorithm', 'dqn')
        self.num_players = config.get('num_players', 4)
        self.no_tasks = config.get('no_tasks', 4)
        self.is_clone = config.get('is_clone', False)
        self.action_recorder = []
        self.timestep = 0
        state_shape_size = self.get_state_shape_size()
        self.state_shape = [[1, state_shape_size] for _ in range(self.num_players)]
        self.action_shape = [[ActionEvent.get_num_actions(self.skip_signals)] for _ in range(self.num_players)]
        self.num_actions = self.get_num_actions()
        self.np_random = np.random.default_rng()

    def clone(self) -> 'CrewEnv':
        new_env = CrewEnv({"num_players": self.num_players,
                           "no_tasks": self.no_tasks,
                           "skip_signals": self.skip_signals,
                           "algorithm": self.algorithm,
                           "is_clone": True})
        new_env.game = self.game.clone()
        new_env.agents = self.agents
        return new_env

    def get_num_actions(self) -> int:
        return ActionEvent.get_num_actions(self.skip_signals)

    def get_player_id(self) -> int:
        return self.game.get_current_player_id()

    def is_over(self) -> bool:
        return self.game.is_over()

    # stub implementation
    def get_state(self, player_id: int):
        return {}

    def set_eval_params(self, eval_mode: bool, eval_hand_id: int = 0):
        self.eval_mode = eval_mode
        self.eval_hand_id = eval_hand_id

    def get_payoffs(self):
        all_tasks_taken = True
        for task in self.game.tasks:
            if not task.taken or task.owner != task.taker:
                all_tasks_taken = False
                break
        payoffs = self.game.payoffs
        if self.algorithm == 'dqn':
            for player_payoff in payoffs:
                if all_tasks_taken:
                    player_payoff[-1] = 1
                else:
                    player_payoff[-1] = -1
        elif all_tasks_taken:
            payoffs = [1 for _ in range(self.num_players)]
        else:
            payoffs = [-1 for _ in range(self.num_players)]
        return payoffs

    def get_state_shape_size(self) -> int:
        state_shape_size = 0
        state_shape_size += 40  # hand_rep_size
        state_shape_size += 4 * 14  # trick_rep_size
        state_shape_size += 4 * 14  # tasks_rep_size
        state_shape_size += 40  # hidden_cards_rep_size
        state_shape_size += 4  # current_player_rep_size
        if not self.skip_signals:
            state_shape_size += 4 * 17  # signal_rep_size
        return state_shape_size

    def _extract_state(self, player_id: int = None):
        extracted_state = {}
        legal_actions = self._get_legal_actions()
        current_player_id = player_id if player_id else self.get_player_id()

        hand_rep = np.zeros(40, dtype=int)
        if not self.is_over():
            for card in self.game.players[current_player_id].hand:
                hand_rep[card.card_id] = 1

        tasks_rep = [np.zeros(14, dtype=int) for _ in range(4)]
        if not self.is_over():
            for task in self.game.tasks:
                tasks_rep[task.owner][task.card.rank_index] = 1
                tasks_rep[task.owner][9 + task.card.suit_index] = 1
                tasks_rep[task.owner][13] = 1 if task.taken else 0

        hidden_cards_rep = np.zeros(40, dtype=int)
        if not self.is_over():
            for player in self.game.players:
                if player.player_id != current_player_id:
                    for card in player.hand:
                        hidden_cards_rep[card.card_id] = 1
                    if player.signal is not None:
                        hidden_cards_rep[player.signal[0].card_id] = 0

        trick_pile_rep = [np.zeros(14, dtype=int) for _ in range(4)]
        if not self.is_over():
            trick_moves = self.game.get_trick_moves()
            for move in trick_moves:
                player_id = move.player_id
                card = move.card
                trick_pile_rep[player_id][card.rank_index] = 1
                trick_pile_rep[player_id][9 + card.suit_index] = 1
                hidden_cards_rep[card.card_id] = 0

        current_player_rep = np.zeros(4, dtype=int)
        current_player_rep[current_player_id] = 1

        rep = []
        rep.append(hand_rep)
        rep += trick_pile_rep
        rep += tasks_rep
        rep.append(hidden_cards_rep)
        rep.append(current_player_rep)

        if not self.skip_signals:
            signals_rep = [np.zeros(17, dtype=int) for _ in range(4)]
            if not self.is_over():
                for player in self.game.players:
                    if player.signal is not None:
                        card_rank = player.signal[0].rank_index
                        card_suit = player.signal[0].suit_index
                        signal_type = player.signal[1].value
                        signals_rep[player.player_id][card_rank] = 1
                        signals_rep[player.player_id][9 + card_suit] = 1
                        signals_rep[player.player_id][14 + signal_type] = 1
            rep += signals_rep
        obs = np.concatenate(rep)
        extracted_state['obs'] = obs
        extracted_state['legal_actions'] = legal_actions
        return extracted_state

    def _decode_action(self, action_id: int):
        return ActionEvent.from_action_id(action_id=action_id)

    def _get_legal_actions(self):
        legal_actions = Judger.get_legal_actions(self.game)
        legal_actions_ids = [action_event.action_id for action_event in legal_actions]
        return legal_actions_ids

    def reset(self):
        state, player_id = self.init_game()
        self.action_recorder = []
        return self._extract_state(), player_id

    def init_game(self):
        self.game = Game(
            num_players=self.num_players,
            no_tasks=self.no_tasks,
            np_random=self.np_random,
            skip_signals=self.skip_signals)
        self.game.init_game()
        current_player_id = self.get_player_id()
        return {}, current_player_id

    def step(self, action_id: int):
        if not self.is_clone:
            self.agents[0].log_move(self.get_player_id(), action_id)
        action = self._decode_action(action_id)
        self.timestep += 1
        self.action_recorder.append((self.get_player_id(), action))
        if isinstance(action, ChooseTaskAction):
            self.game.choose_task(action=action)
        elif isinstance(action, SignalAction) or isinstance(action, SkipSignalAction):
            self.game.signal(action=action)
        else:
            self.game.play_card(action=action)
        next_player_id = self.get_player_id()

        return self._extract_state(), next_player_id

    def set_agents(self, agents):
        self.agents = agents

    def run(self, is_training=False):
        trajectories = [[] for _ in range(self.num_players)]
        state, player_id = self.reset()

        # Loop to play the game
        trajectories[player_id].append(state)
        while not self.game.is_over():
            # Agent plays
            if not is_training:
                action_id, _ = self.agents[player_id].eval_step(state)
            else:
                action_id = self.agents[player_id].step(state)

            # Environment steps
            next_state, next_player_id = self.step(action_id)
            # Save action
            trajectories[player_id].append(action_id)

            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.game.is_over():
                trajectories[player_id].append(state)

        # Add a final state to all the players
        for player_id in range(self.num_players):
            state = self._extract_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()

        return trajectories, payoffs

    def get_action_feature(self, action):
        feature = np.zeros(self.num_actions, dtype=np.int8)
        feature[action] = 1
        return feature

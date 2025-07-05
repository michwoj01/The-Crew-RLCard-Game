import numpy as np

from action_event import ActionEvent
from game import CrewGame
from src.main.judger import Judger
from src.rlcard.envs import Env


class CrewEnv(Env):
    def __init__(self, config):
        self.name = 'crew'
        self.skip_signals = config['skip_signals'] if 'skip_signals' in config else False
        self.algorithm = config['algorithm']
        game = CrewGame(config['num_players'], config['no_tasks'], config['seed'], config['fixed_tasks'],
                        self.skip_signals)
        self.judger: Judger = Judger(game=game)
        super().__init__(game=game, config=config)
        self.state_shape = [(40, 11 if self.skip_signals else 15) for _ in range(config['num_players'])]
        self.action_shape = [[ActionEvent.get_num_actions(self.skip_signals)] for _ in range(config['num_players'])]

    def get_payoffs(self):
        game = self.game
        all_tasks_taken = True
        for task in game.round.tasks:
            if not task.taken or task.owner != task.taker:
                all_tasks_taken = False
                break
        if self.algorithm != 'dmc':
            payoffs = game.round.payoffs
            if self.algorithm == 'dqn':
                for player_payoff in payoffs:
                    if all_tasks_taken:
                        player_payoff[-1] = 1
                    else:
                        player_payoff[-1] = -1
        elif all_tasks_taken:
            payoffs = [1 for _ in range(game.get_num_players())]
        else:
            payoffs = [-1 for _ in range(game.get_num_players())]
        return payoffs

    def _extract_state(self, state):
        game = self.game
        extracted_state = {}
        legal_actions = self.get_legal_actions()
        current_player_id = game.get_player_id()

        obs = [[0 for _ in range(11 if self.skip_signals else 15)] for _ in range(40)]

        if not game.is_over():
            for card in game.round.players[current_player_id].hand:
                obs[card.card_id][0] = 1
            trick_moves = game.round.get_trick_moves()
            if len(trick_moves) > 0:
                first_card_suit = trick_moves[0].card.suit_index
                if first_card_suit == 4:
                    for i in range(4):
                        obs[36 + i][1] = 1
                else:
                    for i in range(9):
                        obs[first_card_suit * 9 + i][1] = 1
            for ind, val in enumerate(game.round.card_record):
                if val > 0:
                    obs[ind][2] = 1
            for move in trick_moves:
                player_id = move.player_id
                card_id = move.card.card_id
                obs[card_id][3 + player_id] = 1
            for task in game.round.tasks:
                player_id = task.owner
                card_id = task.card.card_id
                obs[card_id][7 + player_id] = 1
            if not self.skip_signals:
                for player in game.round.players:
                    if player.signal:
                        card_id = player.signal[0].card_id
                        obs[card_id][11 + player.player_id] = 1

        extracted_state['obs'] = np.array(obs, dtype=np.float32)
        extracted_state['legal_actions'] = legal_actions
        return extracted_state

    def _decode_action(self, action_id: int):
        return ActionEvent.from_action_id(action_id=action_id)

    def get_legal_actions(self):
        legal_actions = self.judger.get_legal_actions()
        legal_actions_ids = [action_event.action_id for action_event in legal_actions]
        return legal_actions_ids

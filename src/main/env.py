import numpy as np

from action_event import ActionEvent
from game import CrewGame
from src.main.card import CrewCard
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
        self.state_shape = [(40, 15 if self.skip_signals else 19) for _ in range(config['num_players'])]
        self.action_shape = [[ActionEvent.get_num_actions(self.skip_signals)] for _ in range(config['num_players'])]

    def get_payoffs(self):
        game = self.game
        all_tasks_taken = True
        for task in game.round.tasks:
            if not task.taken or task.owner != task.taker:
                all_tasks_taken = False
                break
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
        legal_actions = self._get_legal_actions()
        current_player_id = game.get_player_id()
        deck = CrewCard.get_deck()

        obs = [[0 for _ in range(15 if self.skip_signals else 19)] for _ in range(40)]

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
            # META
            for card in deck:  # weak / medium / strong
                strength = (card.rank - 1) // 3
                if card.suit == CrewCard.trump_suit:
                    strength = 3
                obs[card.card_id][11 + strength] = 1
            if len(trick_moves) > 0:  # can take
                first_card = trick_moves[0].card
                for card in deck:
                    if (card.suit == first_card.suit and card.rank > first_card.rank) or \
                            (card.suit == CrewCard.trump_suit and first_card.suit != CrewCard.trump_suit):
                        obs[card.card_id][14] = 1
            # END
            if not self.skip_signals:
                for player in game.round.players:
                    if player.signal:
                        card_id = player.signal[0].card_id
                        obs[card_id][15 + player.player_id] = 1

        extracted_state['obs'] = np.array(obs, dtype=np.float32)
        extracted_state['legal_actions'] = legal_actions
        return extracted_state

    def _decode_action(self, action_id: int):
        return ActionEvent.from_action_id(action_id=action_id)

    def _get_legal_actions(self):
        legal_actions = self.judger.get_legal_actions()
        legal_actions_ids = [action_event.action_id for action_event in legal_actions]
        return legal_actions_ids

    def clone(self) -> 'CrewEnv':
        new_env = CrewEnv({"num_players": self.game.get_num_players(),
                           "no_tasks": self.game.no_tasks,
                           "seed": 42,
                           "fixed_tasks": self.game.fixed_tasks,
                           "skip_signals": self.skip_signals,
                           "algorithm": self.algorithm,
                           "allow_step_back": self.game.allow_step_back})
        new_env.game = self.game.clone()
        new_env.judger = Judger(game=new_env.game)
        return new_env

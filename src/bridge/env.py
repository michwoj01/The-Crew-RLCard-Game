import numpy as np
from collections import OrderedDict
from game import CrewGame
from rlcard.envs import Env
from action_event import ActionEvent
from src.bridge.judger import Judger


class CrewEnv(Env):
    def __init__(self, config):
        self.name = 'crew'
        self.skip_signals = config['skip_signals'] if 'skip_signals' in config else False
        self.game = CrewGame(config['num_players'], config['no_tasks'], config['seed'], config['fixed_tasks'], self.skip_signals)
        self.judger: Judger = Judger(game=self.game)
        super().__init__(config=config)
        state_shape_size = self.get_state_shape_size()
        self.state_shape = [[1, state_shape_size]
                            for _ in range(self.num_players)]

    def get_payoffs(self):
        game = self.game
        payoffs = game.round.payoffs
        all_tasks_taken = True
        for task in game.round.tasks:
            if not task.taken or task.owner != task.taker:
                all_tasks_taken = False
                break
        for player_payoff in payoffs:
            if all_tasks_taken:
                player_payoff[-1] = 1
            else:
                player_payoff[-1] = -1
        return payoffs

    def get_state_shape_size(self) -> int:
        state_shape_size = 0
        state_shape_size += 40  # hand_rep_size
        state_shape_size += 4 * 14  # trick_rep_size
        state_shape_size += 4 * 13  # tasks_rep_size
        state_shape_size += 40  # hidden_cards_rep_size
        state_shape_size += 4  # current_player_rep_size
        if not self.skip_signals:
            state_shape_size += 4 * 17  # signal_rep_size
        return state_shape_size

    def _extract_state(self, state):
        game = self.game
        extracted_state = {}
        legal_actions: OrderedDict = self.get_legal_actions()
        raw_legal_actions = list(legal_actions.keys())
        current_player_id = game.get_player_id()

        hand_rep = np.zeros(40, dtype=int)
        if not game.is_over():
            for card in game.round.players[current_player_id].hand:
                hand_rep[card.card_id] = 1

        tasks_rep = [np.zeros(13, dtype=int) for _ in range(4)]
        if not game.is_over():
            for task in game.round.tasks:
                tasks_rep[task.owner][task.card.rank_index] = 1
                tasks_rep[task.owner][9 + task.card.suit_index] = 1


        hidden_cards_rep = np.zeros(40, dtype=int)
        if not game.is_over():
            for player in game.round.players:
                if player.player_id != current_player_id:
                    for card in player.hand:
                        hidden_cards_rep[card.card_id] = 1
                    if player.signal is not None:
                        hidden_cards_rep[player.signal[0].card_id] = 0

        trick_pile_rep = [np.zeros(14, dtype=int) for _ in range(4)]
        if not game.is_over():
            trick_moves = game.round.get_trick_moves()
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
            if not game.is_over():
                for player in game.round.players:
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
        extracted_state['raw_legal_actions'] = raw_legal_actions
        return extracted_state

    def _decode_action(self, action_id):
        return ActionEvent.from_action_id(action_id=action_id)

    def get_legal_actions(self):
        legal_actions = self.judger.get_legal_actions()
        legal_actions_ids = {action_event.action_id: None for action_event in legal_actions}
        return OrderedDict(legal_actions_ids)

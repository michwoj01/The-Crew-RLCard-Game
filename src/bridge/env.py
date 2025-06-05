import numpy as np
from collections import OrderedDict
from game import CrewGame
from rlcard.envs import Env
from action_event import ActionEvent


class CrewEnv(Env):
    def __init__(self, config):
        self.name = 'crew'
        self.game = CrewGame()
        super().__init__(config=config)
        self.crewPayoffDelegate = DefaultCrewPayoffDelegate()
        self.crewStateExtractor = DefaultCrewStateExtractor()
        state_shape_size = self.crewStateExtractor.get_state_shape_size()
        self.state_shape = [[1, state_shape_size] for _ in range(self.num_players)]

    def get_payoffs(self):
        return self.crewPayoffDelegate.get_payoffs(game=self.game)

    def get_perfect_information(self):
        return self.game.round.get_perfect_information()

    def _extract_state(self, state):  # wch: don't use state 211126
        return self.crewStateExtractor.extract_state(game=self.game)

    def _decode_action(self, action_id):
        return ActionEvent.from_action_id(action_id=action_id)


class CrewPayoffDelegate(object):

    def get_payoffs(self, game: CrewGame):
        raise NotImplementedError


class DefaultCrewPayoffDelegate(CrewPayoffDelegate):

    def get_payoffs(self, game: CrewGame):
        x = sum([len(player.tasks_assigned) for player in game.round.players])
        y = sum([len(player.tasks_completed) for player in game.round.players])
        payoffs = [(y / (x + y)) if len(player.tasks_completed) == 0 else 1.2 * (y / (x + y)) for player in game.round.players]
        return np.array(payoffs)


class CrewStateExtractor(object):

    def get_state_shape_size(self) -> int:
        raise NotImplementedError

    def extract_state(self, game: CrewGame):
        raise NotImplementedError

    @staticmethod
    def get_legal_actions(game: CrewGame):
        legal_actions = game.judger.get_legal_actions()
        legal_actions_ids = {action_event.action_id: None for action_event in legal_actions}
        return OrderedDict(legal_actions_ids)


class DefaultCrewStateExtractor(CrewStateExtractor):

    def get_state_shape_size(self) -> int:
        state_shape_size = 0
        state_shape_size += 4 * 40  # hands_rep_size
        state_shape_size += 4 * 40  # trick_rep_size
        state_shape_size += 40      # hidden_cards_rep_size
        state_shape_size += 4 * 120 # signal_rep_size
        state_shape_size += 4 * 36  # tasks_rep_size
        state_shape_size += 4  # current_player_rep_size
        return state_shape_size

    def extract_state(self, game: CrewGame):
        extracted_state = {}
        legal_actions: OrderedDict = self.get_legal_actions(game=game)
        raw_legal_actions = list(legal_actions.keys())
        current_player = game.round.get_current_player()
        current_player_id = current_player.player_id

        # construct hands_rep of hands of players
        hands_rep = [np.zeros(40, dtype=int) for _ in range(4)]
        if not game.is_over():
            for card in game.round.players[current_player_id].hand:
                hands_rep[current_player_id][card.card_id] = 1

        # construct trick_pile_rep
        trick_pile_rep = [np.zeros(40, dtype=int) for _ in range(4)]
        if not game.is_over():
            trick_moves = game.round.get_trick_moves()
            for move in trick_moves:
                player = move.player
                card = move.card
                trick_pile_rep[player.player_id][card.card_id] = 1

        tasks_rep = [np.zeros(36, dtype=int) for _ in range(4)]
        if not game.is_over():
            for player in game.round.players:
                for card in player.tasks_assigned:
                    tasks_rep[player.player_id][card.card_id] = 1

        signals_rep = [np.zeros(120, dtype=int) for _ in range(4)]
        if not game.is_over():
            for player in game.round.players:
                if player.signal is not None:
                    signals_rep[player.player_id][player.signal[1].value * 40 + player.signal[0].card_id] = 1

        hidden_cards_rep = np.zeros(40, dtype=int)
        if not game.is_over():
            for player in game.round.players:
                if player.player_id != current_player_id:
                    for card in player.hand:
                        hidden_cards_rep[card.card_id] = 1
                    if player.signal is not None:
                        hidden_cards_rep[player.signal[0].card_id] = 0

        # construct current_player_rep
        current_player_rep = np.zeros(4, dtype=int)
        current_player_rep[current_player_id] = 1

        rep = []
        rep += hands_rep
        rep += trick_pile_rep
        rep += tasks_rep
        rep += signals_rep
        rep.append(hidden_cards_rep)
        rep.append(current_player_rep)

        obs = np.concatenate(rep)
        extracted_state['obs'] = obs
        extracted_state['legal_actions'] = legal_actions
        extracted_state['raw_legal_actions'] = raw_legal_actions
        extracted_state['raw_obs'] = obs
        return extracted_state

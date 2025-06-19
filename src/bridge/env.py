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
        self.state_shape = (40, 15)

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

    def _extract_state(self, state):
        game = self.game
        extracted_state = {}
        legal_actions: OrderedDict = self.get_legal_actions()
        raw_legal_actions = list(legal_actions.keys())
        current_player_id = game.get_player_id()

        obs = [[0 for _ in range(15)] for _ in range(40)]

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
            for player in game.round.players:
                if player.signal:
                    card_id = player.signal[0].card_id
                    obs[card_id][11 + player.player_id] = 1

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

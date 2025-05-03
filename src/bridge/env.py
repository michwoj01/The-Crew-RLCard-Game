import numpy as np
from rlcard.envs import Env
from utils.card import CrewCard, Communicate, Signal
from game import CrewGame

class CrewRLCardEnv(Env):
    def __init__(self, config):
        self.game = CrewGame()
        self.name = 'crew'
        super().__init__(config)

    def reset(self) -> tuple[np.ndarray, int]:
        '''
        Returns the beggining state of the first player and his ID
        '''
        state, player = self.game.init_game(self.agents, no_missions=4)
        return self._extract_state(state), player

    def step(self, action, raw_action=False) -> tuple[dict, int]:
        '''
        Takes action taken by the current player
        Returns the next state and the ID of the next player
        '''
        card = self._decode_action(action)
        self.action_recorder.append((self.get_player_id(), card))
        next_state, next_player = self.game.step(card)
        return self._extract_state(next_state), next_player

    def step_back(self) -> tuple[dict, int]:
        raise NotImplementedError

    def is_over(self) -> bool:
        return self.game.game_failed or all(len(player.missions) == 0 for player in self.agents)

    def get_player_id(self) -> int:
        return self.game.current_player

    def get_payoffs(self) -> list[float]:
        return [1 if not player.missions else 0 for player in self.agents]

    def get_perfect_information(self) -> dict:
        return {
            'hand': [[c.to_tuple() for c in player.hand] for player in self.agents],
            'missions': [[c.to_tuple() for c in player.missions] for player in self.agents],
            'signals': [[(c.to_tuple(), signal.value) for c, signal in player.signals] for player in self.agents],
            'tricks': [[(pid, c.to_tuple()) for pid, c in trick] for trick in self.game.tricks],
            'current_player': self.get_player_id(),
            'current_trick': [(pid, c.to_tuple()) for pid, c in self.game.current_trick],
        }

    def _extract_state(self, state: dict) -> dict:
        obs = self._encode_cards(state['hand'])
        raw_legal_actions = self._get_legal_actions(state['hand'])
        legal_action_ids = [self._encode_action(card) for card in raw_legal_actions]
        legal_actions = {action_id: 1.0 for action_id in legal_action_ids}
        return {
            'obs': np.array(obs, dtype=np.int32),
            'legal_actions': legal_actions,
            'raw_legal_actions': legal_action_ids
        }
    
    def _get_legal_actions(self, hand: list[CrewCard]) -> list[CrewCard]:
        if not self.game.current_trick:
            legal_actions = hand
        else:
            leading_suit = self.game.current_trick[0][1].suit
            legal_actions = [
                card for card in hand if card.suit == leading_suit]
            if not legal_actions:
                legal_actions = hand
        return legal_actions
    def _decode_action(self, action: int):
        suit = ['B', 'G', 'Y', 'P', 'R'][action // 9]
        rank = action % 9 + 1
        return CrewCard(suit, rank)

    def _decode_signal(self, action: int) -> Communicate:
        card_code = action // 10
        signal_code = action % 10
        suit_index = card_code // 9
        rank = (card_code % 9) + 1
        card = CrewCard(['B', 'G', 'Y', 'P', 'R'][suit_index], rank)
        signal = Signal(signal_code)
        return card, signal

    # addtional methods

    def _encode_action(self, card: CrewCard) -> int:
        suit_index = ['B', 'G', 'Y', 'P', 'R'].index(card.suit)
        return suit_index * 9 + (card.rank - 1)

    def _encode_cards(self, hand: list[CrewCard]) -> list[int]:
        encoded = [0] * 40
        for card in hand:
            idx = self._encode_action(card)
            encoded[idx] = 1
        return encoded

    def _encode_signals(self, hand: list[Communicate]) -> list[int]:
        encoded = [0] * 443
        for card, signal in hand:
            idx = self._encode_action(card) * 10 + signal.value
            encoded[idx] = 1
        return encoded

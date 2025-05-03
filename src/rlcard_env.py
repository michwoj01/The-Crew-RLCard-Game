import asyncio
import numpy as np
from rlcard.envs import Env
from card import CrewCard
from game import CrewGame
from players import CrewPlayer


class CrewRLCardEnv(Env):
    def __init__(self, config):
        self.game = CrewGame()
        self.name = 'crew'
        super().__init__(config)
        self.players = [CrewPlayer(i) for i in range(4)]
        self.game_failed = False

    def reset(self) -> tuple[np.ndarray, int]:
        '''
        Returns the beggining state of the first player and his ID
        '''
        self.game.init_game(self.players, no_missions=4)
        player = self.game.current_player
        return self._extract_state(None, player), player

    def step(self, action) -> tuple[dict, int]:
        '''
        Takes action taken by the current player
        Returns the next state and the ID of the next player
        '''
        raise NotImplementedError
    
    def step_back(self) -> tuple[dict, int]:
        raise NotImplementedError

    def is_over(self) -> bool:
        return self.game_failed or all(len(player.missions) == 0 for player in self.players)

    def get_player_id(self) -> int:
        return self.game.current_player

    def get_payoffs(self) -> list[float]:
        return [1 if not player.missions else 0 for player in self.players]

    def get_perfect_information(self) -> dict:
        return {
            'hand': [[c.to_tuple() for c in player.hand] for player in self.players],
            'missions': [[c.to_tuple() for c in player.missions] for player in self.players],
            'signals': [[(c.to_tuple(), signal.value) for c, signal in player.signals] for player in self.players],
            'tricks': [[(pid, c.to_tuple()) for pid, c in trick] for trick in self.game.tricks],
            'current_player': self.get_player_id(),
            'current_trick': [(pid, c.to_tuple()) for pid, c in self.game.current_trick],
        }

    def _extract_state(self, state: dict) -> np.ndarray:
        raise NotImplementedError

    def _decode_action(self, action: int):
        suit = ['B', 'G', 'Y', 'P', 'R'][action // 9]
        rank = action % 9 + 1
        return CrewCard(suit, rank)

    def _get_legal_actions(self, player_id) -> dict:
        return {
            self._encode_action(card): None for card in self.players[player_id].hand
        }

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

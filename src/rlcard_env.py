import asyncio
import numpy as np
from rlcard.envs import Env
from card import CrewCard
from game import CrewGame
from players import CrewPlayer


class CrewRLCardEnv(Env):
    def __init__(self, config):
        super().__init__(config)
        self.game = CrewGame()
        self.players = [CrewPlayer(i) for i in range(4)]
        self.game_failed = False

    def get_payoffs(self):
        return [1 if not player.missions else 0 for player in self.players]

    def get_perfect_information(self):
        return {
            'hand': [[c.to_tuple() for c in player.hand] for player in self.players],
            'missions': [[c.to_tuple() for c in player.missions] for player in self.players],
            'signals': [[(c.to_tuple(), signal.value) for c, signal in player.signals] for player in self.players],
            'tricks': [[(pid, c.to_tuple()) for pid, c in trick] for trick in self.game.tricks],
            'current_player': self.get_player_id(),
            'current_trick': [(pid, c.to_tuple()) for pid, c in self.game.current_trick],
        }

    def reset(self) -> dict:
        self.game.init_game(self.players, no_missions=4)
        self.game_failed = False
        self.game.tricks = []
        self.state = self._extract_state(
            self.game.get_state(self.get_curr_player_id()))
        return self.state

    def step(self, action):
        decoded_action = self._decode_action(action)
        game_over = self.game.step(decoded_action)

        self.state = self.get_state(self.get_curr_player())
        reward = self._get_reward()
        done = self._is_done()
        return self.state, reward, done, {}

    def _extract_state(self, state=None) -> dict:
        obs = self._encode_cards(self.get_curr_player().hand)
        return {
            'obs': np.array(obs),
            'legal_actions': self._get_legal_actions()
        }

    def _get_legal_actions(self) -> dict:
        return {
            self._encode_action(card): None for card in self.get_curr_player().hand
        }

    def _decode_action(self, action):
        suit = ['B', 'G', 'Y', 'P', 'R'][action // 9]
        rank = action % 9 + 1
        return CrewCard(suit, rank)

    def _encode_action(self, card: CrewCard) -> int:
        suit_index = ['B', 'G', 'Y', 'P', 'R'].index(card.suit)
        return suit_index * 9 + (card.rank - 1)

    def _encode_cards(self, hand: list[CrewCard]) -> list[int]:
        encoded = [0] * 40
        for card in hand:
            idx = self._encode_action(card)
            encoded[idx] = 1
        return encoded

    def _get_reward(self):
        if self.game_failed:
            return -1
        elif all(len(player.missions) == 0 for player in self.players):
            return 1
        return 0

    def _is_done(self):
        return self.game_failed or all(len(player.missions) == 0 for player in self.players)
    
    def get_curr_player_id(self) -> int:
        return self.game.current_player

    def get_curr_player(self) -> CrewPlayer:
        return self.players[self.game.current_player]

    def get_state(self, player_id):
        return self._extract_state()

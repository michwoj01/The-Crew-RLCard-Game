import asyncio
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
            'current_player': self.game.current_player,
            'current_trick': [(pid, c.to_tuple()) for pid, c in self.game.current_trick],
        }

    def get_player_id(self):
        return self.game.current_player

    def get_state(self, player_id):
        return self._extract_state(self.game.get_state(player_id))

    def reset(self) -> dict:
        asyncio.run(self.game.init_game(self.players, no_missions=4))
        self.game_failed = False
        self.game.tricks = []
        self.state = self._extract_state()
        return self.state

    def step(self, action) -> tuple:
        player: CrewPlayer = self.players[self.game.current_player]
        card_to_play = player.play_card(self.game.current_trick)

        if card_to_play not in player.hand:
            raise ValueError("Invalid action: card not in hand")

        player.hand.remove(card_to_play)
        self.game.current_trick.append((player.player_id, card_to_play))

        # Update leading suit if first card
        if len(self.game.current_trick) == 1:
            self.game.leading_suit = card_to_play.suit

        self.game.current_player = (self.game.current_player + 1) % 4

        if len(self.game.current_trick) == 4:
            result, msg = self.game._resolve_winner(len(self.game.tricks))
            self.game.tricks.append(self.game.current_trick)
            self.game.current_trick = []
            self.game.leading_suit = None
            if not result:
                self.game_failed = True

        self.state = self.get_state(self.game.current_player)
        reward = self._get_reward()
        done = self._is_done()
        return self.state, reward, done, {}

    def _extract_state(self, state) -> dict:
        current_player: CrewPlayer = self.game.players[self.game.current_player]
        state = {
            'obs': self._encode_cards(current_player.hand),
            'legal_actions': self._get_legal_actions()
        }
        return state

    def _get_legal_actions(self) -> dict:
        current_player = self.game.players[self.game.current_player]
        return {
            self._encode_action(card): None for card in current_player.hand
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

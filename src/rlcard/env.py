import numpy as np
from rlcard.envs import Env
from utils.card import CrewCard, Communicate, Signal
from game import CrewGame
from utils.syntax_sugar import overrides

class CrewEnv(Env):
    def __init__(self, config):
        self.game = CrewGame()
        self.name = 'crew'
        super().__init__(config)

    @overrides(Env)
    def reset(self):
        state, player = self.game.init_game(self.agents, no_tasks=4)
        self.action_recorder = []
        return self._extract_state(state), player

    @overrides(Env)
    def is_over(self) -> bool:
        return self.game.is_over()

    @overrides(Env)
    def get_player_id(self) -> int:
        return self.game.current_player

    # check payoffs
    @overrides(Env)
    def get_payoffs(self) -> list[float]:
        return [1 if not player.tasks_assigned else 0 for player in self.agents]

# maybe to delete
    @overrides(Env)
    def get_perfect_information(self) -> dict:
        return {
            'hand': [[c.to_tuple() for c in player.hand] for player in self.agents],
            'tasks': [[c.to_tuple() for c in player.tasks_assigned] for player in self.agents],
            'tricks': [[(pid, c.to_tuple()) for pid, c in trick] for trick in self.game.tricks],
            'current_player': self.get_player_id(),
            'current_trick': [(pid, c.to_tuple()) for pid, c in self.game.current_trick],
        }

    def _extract_state(self, state: dict) -> dict:
        obs = self._encode_actions(state['hand'])
        raw_legal_actions = self._get_legal_actions(state['hand'])
        legal_action_ids = [self._encode_action(
            card) for card in raw_legal_actions]
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

        # if not self.game.players[self.get_player_id()].has_communicated:
        #     signals = []
        #     for suit in ['B', 'G', 'Y', 'P', 'R']:
        #         cards_of_suit = [card for card in hand if card.suit == suit]
        #         if cards_of_suit:
        #             if len(cards_of_suit) == 1:
        #                 signals.append(Communicate(
        #                     cards_of_suit[0], Signal.ONLY))
        #             else:
        #                 signals.append(Communicate(
        #                     min(cards_of_suit, key=lambda c: c.rank), Signal.LOWEST))
        #                 signals.append(Communicate(
        #                     max(cards_of_suit, key=lambda c: c.rank), Signal.HIGHEST))
        #     legal_actions.extend(signals)
        return legal_actions

    def _decode_action(self, action: int) -> CrewCard:
        if action < 40:
            suit = ['B', 'G', 'Y', 'P', 'R'][action // 9]
            rank = action % 9 + 1
            return CrewCard(suit, rank)
        else:
            return self._decode_signal(action)

    def _decode_signal(self, action: int):
        card_code = action // 40
        signal_code = action % 40
        suit_index = card_code // 9
        rank = (card_code % 9) + 1
        card = CrewCard(['B', 'G', 'Y', 'P', 'R'][suit_index], rank)
        signal = Signal(signal_code)
        return card, signal

    def _encode_action(self, card: CrewCard) -> int:
        suit_index = ['B', 'G', 'Y', 'P', 'R'].index(card.suit)
        return suit_index * 9 + (card.rank - 1)

    def _encode_signal_action(self, card: CrewCard, signal_type: Signal) -> int:
        base = self._encode_action(card)
        if signal_type == Signal.LOWEST:
            return 40 + base
        elif signal_type == Signal.HIGHEST:
            return 80 + base
        elif signal_type == Signal.ONLY:
            return 120 + base
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

    def _encode_actions(self, hand: list[CrewCard], signals: list[Communicate] = None) -> list[int]:
        encoded = [0] * 40
        for card in hand:
            idx = self._encode_action(card)
            encoded[idx] = 1
        # for card, signal in signals:
        #     idx = self._encode_signal_action(card, signal)
        #     encoded[idx] = 1
        return encoded

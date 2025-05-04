import numpy as np
from rlcard.envs import Env
from utils.card import CrewCard, Communicate, Signal
from game import CrewGame


class CrewEnv(Env):
    def __init__(self, config):
        self.game = CrewGame()
        self.name = 'crew'
        super().__init__(config)

    def run(self, is_training=False):
        self.reset()
        trajectories = [[] for _ in range(len(self.agents))]
        player_id = self.get_player_id()
        state = self._extract_state(self.game.get_state(player_id))

        while not self.is_over():
            if is_training:
                action = self.agents[player_id].step(state)
            else:
                action, _ = self.agents[player_id].eval_step(state)
            
            next_state, next_player_id = self.step(action)
            reward = 0 
            done = self.is_over()

            trajectories[player_id].append((state, action, reward, next_state, done))

            state = next_state
            player_id = next_player_id

        payoffs = self.get_payoffs()

        for i in range(len(self.agents)):
            if trajectories[i]:
                last = trajectories[i][-1]
                trajectories[i][-1] = (last[0], last[1], payoffs[i], last[3], True)

        return trajectories, payoffs


    def reset(self) -> tuple[np.ndarray, int]:
        state, player = self.game.init_game(self.agents, no_missions=4)
        return self._extract_state(state), player

    def step(self, action: int, raw_action=False) -> tuple[dict, int]:
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

    def _decode_signal(self, action: int) -> Communicate:
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

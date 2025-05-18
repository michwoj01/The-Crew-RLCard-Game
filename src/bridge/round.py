from typing import List
from dealer import Dealer
from players import CrewPlayer
from action_event import ActionEvent
from card import CrewCard, Signal

class Round:

    def __init__(self, num_players: int, np_random):
        self.np_random = np_random
        self.dealer: Dealer = Dealer(self.np_random)
        self.players: List[CrewPlayer] = []
        for player_id in range(num_players):
            self.players.append(CrewPlayer(
                player_id=player_id, np_random=self.np_random))
        self.current_player_id: int = 0
        self.play_card_count: int = 0
        self.round_number: int = 0
        self.signals_used: List[bool] = [False] * num_players

    def is_over(self) -> bool:
        return self.round_number >= 10 or all(len(player.tasks) == 0 for player in self.players)

    def get_current_player(self) -> CrewPlayer:
        return self.players[self.current_player_id]

    def play_card(self, action: ActionEvent):
        current_player = self.players[self.current_player_id]
        if ActionEvent.is_play_card_action(action.action_id):
            card = action.card
            current_player.remove_card_from_hand(card=card)
            self.play_card_count += 1
            if self.play_card_count % 4 == 0:
                self.round_number += 1
            self.current_player_id = (self.current_player_id + 1) % 4
        elif ActionEvent.is_signal_card_action(action.action_id):
            if self.signals_used[self.current_player_id]:
                raise ValueError("Signal already used by this player")
            self.signals_used[self.current_player_id] = True
            # Handle signal logic here
        else:
            raise ValueError("Invalid action")

    def get_legal_actions(self) -> List[ActionEvent]:
        current_player = self.players[self.current_player_id]
        legal_actions: List[ActionEvent] = []
        if not self.signals_used[self.current_player_id]:
            legal_actions.extend(self._get_signal_actions(current_player))
        legal_actions.extend(self._get_play_card_actions(current_player))
        return legal_actions

    def _get_signal_actions(self, player: CrewPlayer) -> List[ActionEvent]:
        signals: List[ActionEvent] = []
        for card in player.hand:
            signals.append(ActionEvent.get_action_id_for_signal_card(card, Signal.LOWEST)) 
            signals.append(ActionEvent.get_action_id_for_signal_card(card, Signal.HIGHEST))
            signals.append(ActionEvent.get_action_id_for_signal_card(card, Signal.ONLY))
        return signals

    def _get_play_card_actions(self, player: CrewPlayer) -> List[ActionEvent]:
        return [ActionEvent.get_action_id_for_play_card(card) for card in player.hand]

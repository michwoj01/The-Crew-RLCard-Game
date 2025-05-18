from typing import List
from action_event import ActionEvent, PlayCardAction
from card import CrewCard

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game import CrewGame


class Judger:

    def __init__(self, game: 'CrewGame'):
        self.game: CrewGame = game

    def get_legal_actions(self) -> List[ActionEvent]:
        legal_actions: List[ActionEvent] = []
        if not self.game.is_over():
            current_player = self.game.round.get_current_player()
            trick_moves = self.game.round.get_trick_moves()
            hand = self.game.round.players[current_player.player_id].hand
            legal_cards = hand
            if trick_moves and len(trick_moves) < 4:
                led_card: CrewCard = trick_moves[0].card
                cards_of_led_suit = [card for card in hand if card.suit == led_card.suit]
                if cards_of_led_suit:
                    legal_cards = cards_of_led_suit
            for card in legal_cards:
                action = PlayCardAction(card=card)
                legal_actions.append(action)
        return legal_actions

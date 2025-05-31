from typing import List
from action_event import ActionEvent, PlayCardAction, ChooseTaskAction
from card import CrewCard

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game import CrewGame


class Judger:

    def __init__(self, game: 'CrewGame'):
        self.game: CrewGame = game

    def get_legal_actions(self) -> List[ActionEvent]:
        legal_actions: List[ActionEvent] = []
        match self.game.round.round_phase:
            case 'game over':
                legal_actions = []
            case 'choosing tasks':
                for task in self.game.round.dealer.tasks:
                    legal_actions.append(ChooseTaskAction(task=task))
            case 'playing card':
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
                    legal_actions.append(PlayCardAction(card=card))
        return legal_actions

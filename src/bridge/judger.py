from typing import List
from action_event import ActionEvent, PlayCardAction, ChooseTaskAction, SignalAction, SkipSignalAction
from card import CrewCard, SignalType

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
                for card in self.game.round.dealer.tasks:
                    legal_actions.append(ChooseTaskAction(card=card))
            case 'signaling':
                legal_actions.append(SkipSignalAction())
                current_player = self.game.round.get_current_player()
                if current_player.can_signal():
                    hand = current_player.hand
                    suits = {card.suit for card in hand}
                    for suit in suits:
                        cards_of_suit = [
                            card for card in hand if card.suit == suit]
                        if len(cards_of_suit) == 1:
                            legal_actions.append(SignalAction(
                                card=cards_of_suit[0], signal_type=SignalType.ONLY))
                        else:
                            legal_actions.append(
                                SignalAction(card=min(cards_of_suit, key=lambda c: c.rank), signal_type=SignalType.LOWEST))
                            legal_actions.append(
                                SignalAction(card=max(cards_of_suit, key=lambda c: c.rank), signal_type=SignalType.HIGHEST))
            case 'playing card':
                current_player = self.game.round.get_current_player()
                trick_moves = self.game.round.get_trick_moves()
                hand = current_player.hand
                legal_cards = hand
                if trick_moves and len(trick_moves) < 4:
                    led_card: CrewCard = trick_moves[0].card
                    cards_of_led_suit = [
                        card for card in hand if card.suit == led_card.suit]
                    if cards_of_led_suit:
                        legal_cards = cards_of_led_suit
                for card in legal_cards:
                    legal_actions.append(PlayCardAction(card=card))
        return legal_actions

import numpy as np
from utils.card import CrewCard, Communicate, Signal
from rlcard.agents import RandomAgent

class CrewRLCardPlayer(RandomAgent):

    def __init__(self, num_actions, player_id: int):
        super().__init__(num_actions)
        self.player_id: int = player_id
        self.has_communicated: bool = False
        self.hand: list[CrewCard] = []
        self.missions: list[CrewCard] = []
        self.signals: list[Communicate] = []

    def choose_task(self, tasks: list[CrewCard]) -> CrewCard:
        return np.random.choice(tasks)

    def communicate(self) -> tuple[int, Communicate]:
        if self.has_communicated:
            raise ValueError("Player has already communicated")
        signal: Communicate = self.signals[0]
        self.has_communicated = True
        return (self.player_id, signal)

    def complete_mission(self, card: CrewCard):
        if card in self.missions:
            self.missions.remove(card)

    def __str__(self):
        return f'Player {self.player_id}'

    def show_hand_and_task(self):
        print(f'Player {self.player_id} hand: ', [
            card.suit[0] + str(card.rank) for card in self.hand], ' Tasks: ', [
            card.suit[0] + str(card.rank) for card in self.missions])

    def update_possible_communications(self):
        self.signals = []
        suits = {card.suit for card in self.hand}
        for suit in suits:
            cards_of_suit = [card for card in self.hand if card.suit == suit]
            if len(cards_of_suit) == 1:
                self.signals.append((cards_of_suit[0], Signal.ONLY))
            else:
                self.signals.append(
                    (min(cards_of_suit, key=lambda c: c.rank), Signal.LOWEST))
                self.signals.append(
                    (max(cards_of_suit, key=lambda c: c.rank), Signal.HIGHEST))

    def _encode_action(self, card):
        suit_index = ['B', 'G', 'Y', 'P', 'R'].index(card.suit)
        return suit_index * 9 + (card.rank - 1)

    def _decode_action(self, action):
        suit = ['B', 'G', 'Y', 'P', 'R'][action // 9]
        rank = action % 9 + 1
        return CrewCard(suit, rank)

    def _encode_hand(self):
        enc = [0] * 40
        for c in self.hand:
            enc[self._encode_action(c)] = 1
        return enc

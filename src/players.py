import numpy as np
from card import Communicate, CrewCard, Signal


class CrewPlayer:
    def __init__(self, player_id: int):
        self.player_id: int = player_id
        self.has_communicated: bool = False
        self.hand: list[CrewCard] = []
        self.tasks: list[CrewCard] = []
        self.signals: list[Communicate] = []

    def play_card(self) -> CrewCard:
        card = np.random.choice(self.hand)
        self.hand.remove(card)
        return card

    def choose_task(self, tasks: list[CrewCard]) -> CrewCard:
        return np.random.choice(tasks)

    def communicate(self) -> tuple[int, Communicate]:
        if self.has_communicated:
            raise ValueError("Player has already communicated")
        signal: Communicate = self.signals[0]
        self.has_communicated = True
        return (self.player_id, signal)

    def __str__(self):
        return f'Player {self.player_id}'

    def show_hand_and_task(self):
        print(f'Player {self.player_id} hand: ', [
            card.suit[0] + str(card.rank) for card in self.hand], ' Tasks: ', [
            card.suit[0] + str(card.rank) for card in self.tasks])

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


class IntelligentCrewPlayer(CrewPlayer):
    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.played_cards: list[CrewCard] = []

    def update_state(self, tasks: list[tuple[int, CrewCard]], current_round: list[tuple[int, CrewCard]]):
        self.state = {
            'tasks': tasks,
            'current_round': current_round,
            'played_cards': self.played_cards,
            'possible_communications': self.signals,
            'hand': self.hand,
        }

    def play_card(self) -> CrewCard:
        if not self.state:
            raise ValueError("State not initialized for IntelligentCrewPlayer")

        leading_suit = self.state['current_round'][0][1].suit if self.state['current_round'] else None
        if leading_suit is None:
            task_suit = self.tasks[0].suit if self.tasks else None
            legal_actions = [
                card for card in self.hand if card.suit == task_suit] or self.hand
            chosen_card = max(legal_actions, key=lambda card: card.rank)
            self.hand.remove(chosen_card)
            self.played_cards.append(chosen_card)
            return chosen_card
        else:
            legal_actions = [
                card for card in self.hand if card.suit == leading_suit] or self.hand
            chosen_card = min(legal_actions, key=lambda card: card.rank)
            self.hand.remove(chosen_card)
            self.played_cards.append(chosen_card)
            return chosen_card


class HumanCrewPlayer(CrewPlayer):
    def __init__(self, player_id: int):
        super().__init__(player_id)

    def play_card(self) -> CrewCard:
        self.show_hand_and_task()
        played = True
        while (played):
            suit = input('Enter suit: ')
            rank = int(input('Enter rank: '))
            card = CrewCard(suit, rank)
            if card in self.hand:
                played = False
            else:
                print('Invalid card, try again')
        self.hand.remove(card)
        return card

    def choose_task(self, tasks: list[CrewCard]) -> CrewCard:
        self.show_hand_and_task()
        print(f'Tasks to choose: ', [
            card.suit[0] + str(card.rank) for card in tasks])
        chosen = True
        while (chosen):
            suit = input('Enter suit: ')
            rank = int(input('Enter rank: '))
            card = CrewCard(suit, rank)
            if card in tasks:
                chosen = False
            else:
                print('Invalid task, try again')
        return card

    def communicate(self) -> tuple[int, Communicate]:
        self.show_hand_and_task()
        print(f'Player {self.player_id} signals: ', [
              (str(signal[0]), str(signal[1])) for signal in self.signals])
        if self.has_communicated:
            raise ValueError("Player has already communicated")
        chosen = True
        while (chosen):
            suit = input('Enter suit: ')
            rank = int(input('Enter rank: '))
            signal = CrewCard(suit, rank), Signal(
                int(input('Enter signal: ')))
            if signal in self.signals:
                chosen = False
            else:
                print('Invalid signal, try again')
        self.has_communicated = True
        return (self.player_id, signal)

import numpy as np
import asyncio
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
        self.state = None
        self.played_cards: list[CrewCard] = []

    def update_state(self, tasks: list[tuple[int, CrewCard]], current_round: list[tuple[int, CrewCard]]):
        self.state = {
            'tasks': tasks,
            'current_round': current_round,
            'played_cards': self.played_cards,
            'possible_communications': self.signals,
            'hand': self.hand,
        }

    def choose_task(self, tasks: list[CrewCard]) -> CrewCard:
        for task in tasks:
            highest_card = CrewCard(task.suit, 9)
            if highest_card in self.hand:
                return task
        for task in tasks:
            high_card = CrewCard(task.suit, 8)
            if task.rank != 9 and high_card in self.hand and len(list(filter(lambda card: card.suit == task.suit, self.hand))) > 1:
                return task
        return np.random.choice(tasks)

    def play_card(self) -> CrewCard:
        if not self.state:
            raise ValueError("State not initialized for IntelligentCrewPlayer")

        current_round = self.state['current_round']
        leading_suit = current_round[0][1].suit if current_round else None
        mapped_all_tasks = list(map(lambda task: task[1], self.state['tasks']))
        if leading_suit is None:
            task_suit = self.tasks[0].suit if self.tasks else None
            possible_actions = [card for card in self.hand if card.suit == task_suit and (
                card not in mapped_all_tasks or card in self.tasks)]
            if len(possible_actions) > 0:
                chosen_card = max(possible_actions, key=lambda card: card.rank)
            else:
                safe_cards = [
                    card for card in self.hand if not card.is_rocket and card not in mapped_all_tasks] or self.hand
                chosen_card = min(safe_cards, key=lambda card: card.rank)
            self.hand.remove(chosen_card)
            self.played_cards.append(chosen_card)
            return chosen_card
        else:
            leading_player_id = current_round[0][0]
            leading_player_tasks = list(filter(lambda task: task[0] == leading_player_id, self.state['tasks']))
            mapped_leading_player_tasks = list(map(lambda task: task[1], leading_player_tasks))

            mission_card_for_leading_player = next((card for card in mapped_leading_player_tasks if card in self.hand), None)
            if any(card in list(map(lambda card: card[1], current_round)) for card in self.tasks):
                legal_actions_to_take = [card for card in self.hand if card.suit == leading_suit and (card not in mapped_all_tasks or card in self.tasks)] or [card for card in self.hand if card.suit == leading_suit] or self.hand
                maybe_rocket = next((card for card in legal_actions_to_take if card.is_rocket), None)
                if maybe_rocket is not None:
                    chosen_card = maybe_rocket
                else:
                    chosen_card = max(legal_actions_to_take, key=lambda card: card.rank)
            elif mission_card_for_leading_player is not None and current_round[0][1].rank > mission_card_for_leading_player.rank:
                chosen_card = mission_card_for_leading_player
            else:
                legal_actions_to_give_away = [card for card in self.hand if card.suit == leading_suit and card not in mapped_all_tasks] or [card for card in self.hand if not card.is_rocket and card not in mapped_all_tasks] or [card for card in self.hand if card.suit == leading_suit] or self.hand
                chosen_card = min(legal_actions_to_give_away, key=lambda card: card.rank)
            self.hand.remove(chosen_card)
            self.played_cards.append(chosen_card)
            return chosen_card


class HumanCrewPlayer(CrewPlayer):
    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.selected_card: CrewCard = None
        self.selected_task: CrewCard = None
        self.selected_signal: tuple[CrewCard, Signal] = None

    async def play_card(self) -> CrewCard:
        while not self.selected_card or self.selected_card not in self.hand:
            await asyncio.sleep(0.1)  # Wait for a valid card to be set
        card = self.selected_card
        self.hand.remove(card)
        self.selected_card = None  # Reset after playing
        return card

    async def choose_task(self, tasks: list[CrewCard]) -> CrewCard:
        while not self.selected_task or self.selected_task not in tasks:
            await asyncio.sleep(0.1)  # Wait for a valid task to be set
        task = self.selected_task
        self.selected_task = None  # Reset after choosing
        return task

    async def communicate(self) -> tuple[int, Communicate]:
        if self.has_communicated:
            raise ValueError("Player has already communicated")
        while not self.selected_signal or self.selected_signal not in self.signals:
            await asyncio.sleep(0.1)  # Wait for a valid signal to be set
        signal = self.selected_signal
        self.has_communicated = True
        self.selected_signal = None  # Reset after communicating
        return (self.player_id, signal)

    def set_selected_card(self, card: CrewCard):
        """Set the card to be played."""
        self.selected_card = card

    def set_selected_task(self, task: CrewCard):
        """Set the task to be chosen."""
        self.selected_task = task

    def set_selected_signal(self, signal: tuple[CrewCard, Signal]):
        """Set the signal to be communicated."""
        self.selected_signal = signal

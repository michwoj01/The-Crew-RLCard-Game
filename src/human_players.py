from card import Communicate, CrewCard, Signal
from players import CrewPlayer
import numpy as np


class IntelligentCrewPlayer(CrewPlayer):
    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.state = None
        self.played_cards: list[CrewCard] = []

    def update_state(self, tasks: list[tuple[int, CrewCard]]):
        self.state = {
            'tasks': tasks,
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

    def play_card(self, current_round) -> CrewCard:
        if not self.state:
            raise ValueError("State not initialized for IntelligentCrewPlayer")

        leading_suit = current_round[0][1].suit if current_round else None
        mapped_all_tasks = [task[1] for task in self.state['tasks']]
        if leading_suit is None:
            task_suit = self.missions[0].suit if self.missions else None
            possible_actions = [card for card in self.hand if card.suit == task_suit and (
                card not in mapped_all_tasks or card in self.missions)]
            if possible_actions:
                chosen_card = max(possible_actions, key=lambda card: card.rank)
            else:
                safe_cards = [
                    card for card in self.hand if not card.is_rocket and card not in mapped_all_tasks] or self.hand
                chosen_card = min(safe_cards, key=lambda card: card.rank)
        else:
            leading_player_id = current_round[0][0]
            leading_player_tasks = [
                task[1] for task in self.state['tasks'] if task[0] == leading_player_id]
            mission_card_for_leading_player = next(
                (card for card in leading_player_tasks if card in self.hand), None)
            if any(card in [played_card[1] for played_card in current_round] for card in self.missions):
                legal_actions_to_take = [card for card in self.hand if card.suit == leading_suit and (
                    card not in mapped_all_tasks or card in self.missions)] or [card for card in self.hand if card.suit == leading_suit] or self.hand
                maybe_rocket = next(
                    (card for card in legal_actions_to_take if card.is_rocket), None)
                chosen_card = maybe_rocket if maybe_rocket else max(legal_actions_to_take,
                                                                    key=lambda card: card.rank)
            elif mission_card_for_leading_player and current_round[0][1].rank > mission_card_for_leading_player.rank:
                chosen_card = mission_card_for_leading_player
            else:
                legal_actions_to_give_away = [card for card in self.hand if card.suit == leading_suit and card not in mapped_all_tasks] or [
                    card for card in self.hand if not card.is_rocket and card not in mapped_all_tasks] or [card for card in self.hand if card.suit == leading_suit] or self.hand
                chosen_card = min(legal_actions_to_give_away,
                                  key=lambda card: card.rank)
        self.hand.remove(chosen_card)
        self.played_cards.append(chosen_card)
        return chosen_card


class HumanCrewPlayer(CrewPlayer):
    def __init__(self, player_id: int, websocket):
        super().__init__(player_id)
        self.websocket = websocket

    async def play_card(self, current_round) -> CrewCard:
        legal_actions = None
        if not current_round:
            legal_actions = self.hand
        else:
            leading_suit = current_round[0][1].suit
            legal_actions = [
                card for card in self.hand if card.suit == leading_suit]
            if not legal_actions:
                legal_actions = self.hand
        await self.websocket.send_json({"action": "play_card", "hand": [str(card) for card in legal_actions]})
        while True:
            message = await self.websocket.receive_json()
            if message["action"] == "play_card":
                card = CrewCard(message["suit"], message["rank"])
                if card in self.hand:
                    self.hand.remove(card)
                    return card

    async def choose_task(self, tasks: list[CrewCard]) -> CrewCard:
        await self.websocket.send_json({"action": "choose_task", "tasks": [str(task) for task in tasks], "hand": [str(card) for card in self.hand]})
        while True:
            message = await self.websocket.receive_json()
            if message["action"] == "choose_task":
                task = CrewCard(message["suit"], message["rank"])
                if task in tasks:
                    return task

    async def communicate(self) -> tuple[int, Communicate]:
        if self.has_communicated:
            raise ValueError("Player has already communicated")
        await self.websocket.send_json({"action": "communicate", "signals": [(str(card), str(signal)) for card, signal in self.signals]})
        while True:
            message = await self.websocket.receive_json()
            if message["action"] == "communicate":
                card = CrewCard(message["suit"], message["rank"])
                signal = Signal.fromLetter(message["signal"])
                if (card, signal) in self.signals:
                    self.has_communicated = True
                    return self.player_id, (card, signal)

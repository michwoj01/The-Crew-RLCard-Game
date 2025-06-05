from card import CrewCard, SignalType


class CrewPlayer:

    def __init__(self, player_id: int, np_random):
        if player_id < 0 or player_id > 3:
            raise Exception(f'CrewPlayer has invalid player_id: {player_id}')
        self.np_random = np_random
        self.player_id: int = player_id
        self.hand: list[CrewCard] = []
        self.tasks_assigned: list[CrewCard] = []
        self.tasks_completed: list[CrewCard] = []
        self.signal: tuple[CrewCard, SignalType] = None

    def remove_card_from_hand(self, card: CrewCard):
        self.hand.remove(card)

    def assign_task(self, task: CrewCard):
        self.tasks_assigned.append(task)

    def can_signal(self):
        return self.signal is None and len(self.hand) > 0

    def signal_card(self, card: CrewCard, signal_type: SignalType):
        if self.signal is None:
            self.signal = (card, signal_type)

    def complete_task(self, card_moves: list[CrewCard]):
        for card_move in card_moves:
            card = card_move.card
            if card in self.tasks_assigned:
                self.tasks_completed.append(card)
                self.tasks_assigned.remove(card)

    def __str__(self):
        return ['N', 'E', 'S', 'W'][self.player_id]

from card import CrewCard


class CrewPlayer:

    def __init__(self, player_id: int, np_random):
        if player_id < 0 or player_id > 3:
            raise Exception(f'CrewPlayer has invalid player_id: {player_id}')
        self.np_random = np_random
        self.player_id: int = player_id
        self.hand: list[CrewCard] = []
        self.tasks_assigned: list[CrewCard] = []
        self.tasks_completed: list[CrewCard] = []

    def remove_card_from_hand(self, card: CrewCard):
        self.hand.remove(card)

    def complete_task(self, cards: [CrewCard]):
        for card in cards:
            if card in self.tasks_assigned:
                self.tasks_completed.append(card)
                self.tasks_assigned.remove(card)

    def __str__(self):
        return ['N', 'E', 'S', 'W'][self.player_id]

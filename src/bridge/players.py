from card import CrewCard


class CrewPlayer:

    def __init__(self, player_id: int, np_random):
        if player_id < 0 or player_id > 3:
            raise Exception(f'CrewPlayer has invalid player_id: {player_id}')
        self.np_random = np_random
        self.player_id: int = player_id
        self.hand: list[CrewCard] = []
        self.tasks: list[CrewCard] = []

    def remove_card_from_hand(self, card: CrewCard):
        self.hand.remove(card)

    def complete_task(self, card: CrewCard):
        if card in self.tasks:
            self.tasks.remove(card)
        else:
            raise ValueError(f"Card {card} not in tasks")

    def __str__(self):
        return ['N', 'E', 'S', 'W'][self.player_id]

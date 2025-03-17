import numpy as np


class CrewCard:
    def __init__(self, suit: str, rank: int):
        self.suit: str = suit
        self.rank: int = rank
        self.is_rocket: bool = suit == 'Rocket'

    def __str__(self):
        return f'{self.suit} {self.rank}'


class CrewPlayer:
    def __init__(self, player_id: int):
        self.player_id: int = player_id
        self.hand: list[CrewCard] = []
        self.tasks: list[CrewCard] = []

    def play_card(self, card: CrewCard) -> CrewCard:
        if card in self.hand:
            self.hand.remove(card)
            return card
        return None

    def choose_task(self, tasks: list[CrewCard]) -> CrewCard:
        return np.random.choice(tasks)

    def __str__(self):
        return f'Player {self.player_id}'


class CrewGame:
    def __init__(self):
        self.players: list[CrewPlayer] = [CrewPlayer(i) for i in range(4)]
        self.tasks: list[tuple[int, CrewCard]] = []
        self.current_round: list[tuple[int, CrewCard]] = []
        self.leading_suit = None
        self.winner = None
        self.deck: list[CrewCard] = self.generate_deck()
        self.deal_cards()
        self.starting_player: int = self.find_starting_player()
        self.assign_tasks()
        self.current_player = self.starting_player

    def generate_deck(self) -> list[CrewCard]:
        suits = ['Blue', 'Green', 'Yellow', 'Pink']
        deck = [CrewCard(suit, rank)
                for suit in suits for rank in range(1, 10)]
        rockets = [CrewCard('Rocket', rank) for rank in range(1, 5)]
        return deck + rockets

    def deal_cards(self):
        np.random.shuffle(self.deck)
        for i, card in enumerate(self.deck):
            self.players[i % 4].hand.append(card)

    def assign_tasks(self):
        normal_cards = [card for card in self.deck if not card.is_rocket]
        task_cards = np.random.choice(normal_cards, 4, replace=False)
        n = len(task_cards)
        for i in range(n):
            chosen_task = self.players[i].choose_task(task_cards)
            self.tasks.append((i, chosen_task))
            self.players[i].tasks.append(chosen_task)
            task_cards = [card for card in task_cards if card != chosen_task]

        print('Tasks assigned: ', [
              (player.player_id, player.tasks[0].suit, player.tasks[0].rank) for player in self.players])

    def find_starting_player(self) -> int:
        for player in self.players:
            if any(card.is_rocket and card.rank == 4 for card in player.hand):
                return player.player_id
        return 0

    def play_round(self, round_number: int) -> bool:
        self.current_round = []
        self.leading_suit = None
        print(f'Starting round {round_number}')
        for _ in range(4):
            player: CrewPlayer = self.players[self.current_player]
            card_played = player.play_card(np.random.choice(player.hand))
            print(
                f'Player {player.player_id} played {card_played.suit} {card_played.rank}')
            if not self.leading_suit:
                self.leading_suit = card_played.suit
            self.current_round.append((player.player_id, card_played))
            self.current_player = (self.current_player + 1) % 4
        return self.resolve_winner(round_number)

    def resolve_winner(self, round_number: int) -> bool:
        highest_card = None
        winning_player = None
        has_task = False
        supposed_winner = None
        played_cards = [card for _, card in self.current_round]
        for task in self.tasks:
            if task[1] in played_cards:
                if has_task:
                    print(
                        f'Double task in round {round_number}, so someone will not fullfill ')
                    return False
                self.tasks.remove(task)
                supposed_winner = task[0]
                has_task = True
                print(
                    f'Player {supposed_winner} should take this trick to gather {task[1]}')
        for player_id, card in self.current_round:
            if highest_card is None or (card.suit == self.leading_suit and card.rank > highest_card.rank) or (card.is_rocket and not highest_card.is_rocket):
                highest_card = card
                winning_player = player_id
        if supposed_winner is not None and supposed_winner != winning_player:
            print(
                f'Task winner did not win the round {round_number}, so we lost')
            return False
        print(f'Player {winning_player} won round {round_number}')
        self.current_player = winning_player
        return True


# Example usage
game = CrewGame()
for i in range(10):
    if not game.play_round(i):
        print('Game over')
        break
else:
    print('Game finished')

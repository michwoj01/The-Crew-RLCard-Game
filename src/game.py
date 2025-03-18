import numpy as np
import rlcard


class CrewCard:
    def __init__(self, suit: str, rank: int):
        self.suit: str = suit
        self.rank: int = rank
        self.is_rocket: bool = suit == 'Rocket'

    def __str__(self):
        return f'{self.suit}{self.rank}'


class CrewPlayer:
    def __init__(self, player_id: int):
        self.player_id: int = player_id
        self.hand: list[CrewCard] = []
        self.tasks: list[CrewCard] = []

    def play_card(self) -> CrewCard:
        card = np.random.choice(self.hand)
        self.hand.remove(card)
        return card

    def choose_task(self, tasks: list[CrewCard]) -> CrewCard:
        return np.random.choice(tasks)

    def __str__(self):
        return f'Player {self.player_id}'
    
    def show_hand_and_task(self):
        # print suit and rank for each card in hand

        print(f'Player {self.player_id} hand: ', [
                card.suit[0] + str(card.rank) for card in self.hand], ' Tasks: ', [
                card.suit[0] + str(card.rank) for card in self.tasks])

class IntelligentCrewPlayer(CrewPlayer):
    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.played_cards: list[CrewCard] = []  # Track all played cards

    def update_state(self, tasks: list[tuple[int, CrewCard]], current_round: list[tuple[int, CrewCard]]):
        # Update the state with visible tasks, played cards in the current round, and all played cards
        self.state = {
            'tasks': tasks,
            'current_round': current_round,
            'played_cards': self.played_cards,
            'hand': self.hand,
        }

    def play_card(self) -> CrewCard:
        if not self.state:
            raise ValueError("State not initialized for IntelligentCrewPlayer")
        
        # Simplified decision-making: Play the lowest-ranked card of the leading suit, if possible
        leading_suit = self.state['current_round'][0][1].suit if self.state['current_round'] else None
        if leading_suit is None:
            task_suit = self.tasks[0].suit if self.tasks else None
            legal_actions = [card for card in self.hand if card.suit == task_suit] or self.hand
            chosen_card = max(legal_actions, key=lambda card: card.rank)
            self.hand.remove(chosen_card)
            self.played_cards.append(chosen_card)
            return chosen_card
        else:
            legal_actions = [card for card in self.hand if card.suit == leading_suit] or self.hand
            chosen_card = min(legal_actions, key=lambda card: card.rank)
            self.hand.remove(chosen_card)
            self.played_cards.append(chosen_card)
            return chosen_card


class CrewGame:
    def __init__(self):
        self.players: list[CrewPlayer] = [
            IntelligentCrewPlayer(0),  # Example: First player is intelligent
            CrewPlayer(1),
            CrewPlayer(2),
            CrewPlayer(3),
        ]
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
        suits = ['B', 'G', 'Y', 'P']
        deck = [CrewCard(suit, rank)
                for suit in suits for rank in range(1, 10)]
        rockets = [CrewCard('Rocket', rank) for rank in range(1, 5)]
        return deck + rockets

    def deal_cards(self):
        np.random.shuffle(self.deck)
        for i, card in enumerate(self.deck):
            self.players[i % 4].hand.append(card)
        for player in self.players:
            player.hand = sorted(player.hand, key=lambda card: (card.suit, card.rank))

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
                print(f'Player {player.player_id} has the 4 Rocket')
                return player.player_id
        return 0

    def play_round(self, round_number: int) -> bool:
        self.current_round = []
        self.leading_suit = None
        print(f'Starting round {round_number}')
        for player in self.players:
            player.show_hand_and_task()
        for _ in range(4):
            player: CrewPlayer = self.players[self.current_player]
            if isinstance(player, IntelligentCrewPlayer):
                player.update_state(self.tasks, self.current_round)
            card_played = player.play_card()
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
        suppoded_task = None
        played_cards = [card for _, card in self.current_round]
        for player_id, card in self.current_round:
            if highest_card is None or (card.suit == self.leading_suit and card.rank > highest_card.rank) or (card.is_rocket and not highest_card.is_rocket):
                highest_card = card
                winning_player = player_id
        for task in self.tasks:
            if task[1] in played_cards:
                if task[0] == winning_player:
                    self.players[winning_player].tasks.remove(task[1])
                else:
                    print(f'Player {winning_player} take {task[1]} but it should have been Player {task[0]}')
                    return False
        print(f'Player {winning_player} won round {round_number}')
        self.current_player = winning_player
        return True


# Example usage
game = CrewGame()
for i in range(10):
    if not game.play_round(i):
        break
else:
    print('Game finished')

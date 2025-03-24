import numpy as np
from card import Communicate, CrewCard
from players import CrewPlayer, IntelligentCrewPlayer


class CrewGame:
    def __init__(self, show_logs: bool = False):
        self.players: list[CrewPlayer] = [
            IntelligentCrewPlayer(0),
            IntelligentCrewPlayer(1),
            IntelligentCrewPlayer(2),
            IntelligentCrewPlayer(3),
        ]
        self.tasks: list[tuple[int, CrewCard]] = []
        self.current_round: list[tuple[int, CrewCard]] = []
        self.communication_log: list[tuple[int, Communicate]] = []
        self.leading_suit = None
        self.winner = None
        self.deck: list[CrewCard] = self.generate_deck()
        self.deal_cards()
        self.starting_player: int = self.find_starting_player()
        self.assign_tasks(3)
        self.current_player = self.starting_player
        self.show_logs = show_logs
    
    def play_game(self):
        for i in range(10):
            if not self.play_round(i):
                break
        else:
            print('Game finished')

    def generate_deck(self) -> list[CrewCard]:
        suits = ['B', 'G', 'Y', 'P']
        deck = [CrewCard(suit, rank)
                for suit in suits for rank in range(1, 10)]
        rockets = [CrewCard('R', rank) for rank in range(1, 5)]
        return deck + rockets

    def deal_cards(self):
        np.random.shuffle(self.deck)
        for i, card in enumerate(self.deck):
            self.players[i % 4].hand.append(card)
        for player in self.players:
            player.hand = sorted(
                player.hand, key=lambda card: (card.suit, card.rank))
            player.update_possible_communications()

    def assign_tasks(self, no_of_tasks: int):
        normal_cards = [card for card in self.deck if not card.is_rocket]
        task_cards = np.random.choice(normal_cards, no_of_tasks, replace=False)
        picking_order = [(self.starting_player + i) %
                          len(self.players) for i in range(no_of_tasks)]
        for i in picking_order:
            chosen_task = self.players[i].choose_task(task_cards)
            self.tasks.append((i, chosen_task))
            self.players[i].tasks.append(chosen_task)
            task_cards = [card for card in task_cards if card != chosen_task]

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
        if self.show_logs:
            for player in self.players:
                player.show_hand_and_task()
        print("Communication log:", [(log[0], str(
            log[1][0]) + "-" + str(log[1][1])) for log in self.communication_log])
        print("Tasks: ", [(task[0], str(task[1]))
              for task in self.tasks])
        for _ in range(4):
            player: CrewPlayer = self.players[self.current_player]
            if isinstance(player, IntelligentCrewPlayer):
                player.update_state(self.tasks, self.current_round)
            if not player.has_communicated:
                player_communication = player.communicate()
                print(
                    f'Player {player.player_id} communicated {player_communication[1][0]} {player_communication[1][1]}')
                self.communication_log.append(player_communication)
            card_played = player.play_card()
            print(
                f'Player {player.player_id} played {card_played.suit} {card_played.rank}')
            if not self.leading_suit:
                self.leading_suit = card_played.suit
            self.current_round.append((player.player_id, card_played))
            self.current_player = (self.current_player + 1) % 4
        round_result = self.resolve_winner(round_number)
        for player in self.players:
            if not player.has_communicated:
                player.update_possible_communications()
        return round_result

    def resolve_winner(self, round_number: int) -> bool:
        highest_card = None
        winning_player = None
        played_cards = [card for _, card in self.current_round]
        for player_id, card in self.current_round:
            if highest_card is None or (card.suit == self.leading_suit and card.rank > highest_card.rank) or (
                    card.is_rocket and not highest_card.is_rocket):
                highest_card = card
                winning_player = player_id
        for owner, task in self.tasks:
            if task in played_cards:
                if owner == winning_player:
                    self.players[winning_player].tasks.remove(task)
                else:
                    print(
                        f'Player {winning_player} take {task} but it should have been Player {owner}')
                    return False
        print(f'Player {winning_player} won round {round_number}')
        self.current_player = winning_player
        return True

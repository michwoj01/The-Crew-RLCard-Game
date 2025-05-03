import numpy as np
import logging
from utils.card import Communicate, CrewCard
from players import CrewRLCardPlayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("game.log"),
        logging.StreamHandler()
    ]
)


class CrewGame:
    def __init__(self):
        self.players: list[CrewRLCardPlayer] = []
        self.deck: list[CrewCard] = []
        self.tricks: list[list[tuple[int, CrewCard]]] = []
        self.missions: list[tuple[int, CrewCard]] = []
        self.show_hands = False
        self.communication_log: list[tuple[int, Communicate]] = []
        self.leading_suit = None
        self.current_player = None
        self.current_trick: list[tuple[int, CrewCard]] = []
        self.winner = None
        self.game_failed = False

    def init_game(self, players: list[CrewRLCardPlayer], no_missions: int, show_hands: bool = False):
        self.tricks = []
        self.missions = []
        self.current_trick = []
        self.communication_log = []
        self.players = players
        self.show_hands = show_hands
        self.deck = self._initialize_deck()
        self._deal_cards()
        self.current_player = self._find_starting_player()
        self._assign_tasks(no_missions)
        return self.get_state(self.current_player), self.current_player

    def _initialize_deck(self) -> list[CrewCard]:
        suits = ['B', 'G', 'Y', 'P']
        return [CrewCard(suit, rank) for suit in suits for rank in range(1, 10)] + \
               [CrewCard('R', rank) for rank in range(1, 5)]

    def _deal_cards(self):
        np.random.shuffle(self.deck)
        for i, card in enumerate(self.deck):
            self.players[i % 4].hand.append(card)
        for player in self.players:
            player.hand.sort(key=lambda card: (card.suit, card.rank))
            player.update_possible_communications()

    def _assign_tasks(self, no_of_tasks: int):
        task_cards = np.random.choice(
            [card for card in self.deck if not card.is_rocket], no_of_tasks, replace=False)
        for i in [(self.current_player + j) % len(self.players) for j in range(no_of_tasks)]:
            chosen_task = self.players[i].choose_task(task_cards)
            self.missions.append((i, chosen_task))
            self.players[i].missions.append(chosen_task)
            task_cards = [card for card in task_cards if card != chosen_task]

    def _find_starting_player(self) -> int:
        starting_player = next((player.player_id for player in self.players if any(
            card.is_rocket and card.rank == 4 for card in player.hand)), 0)
        logging.info(f'Player {starting_player} has the 4 Rocket')
        return starting_player

    def step(self, card: CrewCard) -> tuple[dict, int]:
        player: CrewRLCardPlayer = self.players[self.current_player]
        if card not in player.hand:
            raise ValueError("Invalid action: card not in hand")
        player.hand.remove(card)
        self.current_trick.append((self.current_player, card))
        if len(self.current_trick) == 1:
            self.leading_suit = card.suit

        self.current_player = (self.current_player + 1) % 4

        if len(self.current_trick) == 4:
            self._resolve_winner(len(self.tricks))
        return self.get_state(self.current_player), self.current_player

    def _resolve_winner(self, round_number: int):
        highest_card = None
        winning_player = None
        for player_id, card in self.current_trick:
            if highest_card is None or (card.suit == self.leading_suit and card.rank > highest_card.rank) or (
                    card.is_rocket and not highest_card.is_rocket):
                highest_card = card
                winning_player = player_id
        played_cards = [card for _, card in self.current_trick]
        for owner, mission in self.missions:
            if mission in played_cards:
                if owner == winning_player:
                    self.players[winning_player].complete_mission(mission)
                else:
                    failure_message = f'Player {winning_player} took {mission}, but it should have been Player {owner}.'
                    logging.error(failure_message)
                    self.game_failed = True
                    return
        logging.info(f'Player {winning_player} won round {round_number}')
        self.current_player = winning_player
        self.tricks.append(self.current_trick)
        self.current_trick = []
        self.leading_suit = None

    def get_state(self, player_id: int) -> dict:
        player: CrewRLCardPlayer = self.players[player_id]
        return {
            'hand': player.hand,
            'missions': player.missions,
            'signals': player.signals,
        }
    
    def is_over(self) -> bool:
        return self.game_failed or all(len(player.missions) == 0 for player in self.players)

    def get_num_players(self): return 4

    def get_num_actions(self): return 40

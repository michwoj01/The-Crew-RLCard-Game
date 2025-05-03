import numpy as np
import logging
from card import Communicate, CrewCard
from players import CrewPlayer

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
        self.players: list[CrewPlayer] = []
        self.deck: list[CrewCard] = []
        self.tricks: list[list[tuple[int, CrewCard]]] = []
        self.missions: list[tuple[int, CrewCard]] = []
        self.show_hands = False
        self.communication_log: list[tuple[int, Communicate]] = []
        self.leading_suit = None
        self.current_player = None
        self.current_trick: list[tuple[int, CrewCard]] = []
        self.winner = None

    def init_game(self, players: list[CrewPlayer], no_missions: int, show_hands: bool = False):
        self.players = players
        self.show_hands = show_hands
        self.deck = self._initialize_deck()
        self._deal_cards()
        self.current_player = self._find_starting_player()
        self._assign_tasks(no_missions)

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
        starting_player = next((player.player_id for player in self.players if any(card.is_rocket and card.rank == 4 for card in player.hand)), 0)
        logging.info(f'Player {starting_player} has the 4 Rocket')
        return starting_player

    def play_game(self):
        for i in range(10):
            if not self._play_round(i):
                break
    
    def step(self, action):
        player: CrewPlayer = self.players[self.current_player]
        card_to_play = action

        if card_to_play not in player.hand:
            raise ValueError("Invalid action: card not in hand")

        player.hand.remove(card_to_play)
        self.current_trick.append((player.player_id, card_to_play))

        if len(self.current_trick) == 1:
            self.leading_suit = card_to_play.suit

        self.current_player = (self.current_player + 1) % 4

        if len(self.current_trick) == 4:
            result, _ = self._resolve_winner(len(self.tricks))
            self.tricks.append(self.current_trick)
            self.current_trick = []
            self.leading_suit = None
            if not result:
                return True

        return False

    def _play_round(self, round_number: int) -> bool:
        self.current_trick = []
        self.leading_suit = None
        logging.info(f'Starting round {round_number}')
        if self.show_hands:
            [player.show_hand_and_task() for player in self.players]
        logging.info("Communication log: %s", [(log[0], f"{log[1][0]}-{log[1][1]}") for log in self.communication_log])
        logging.info("Missions: %s", [(mission[0], str(mission[1])) for mission in self.missions])
        for _ in range(4):
            player: CrewPlayer = self.players[self.current_player]
            if not player.has_communicated:
                communicate = player.communicate()
                logging.info(f'Player {player.player_id} communicated {communicate[1][0]} {communicate[1][1]}')
                self.communication_log.append(communicate)
            card_played = player.play_card(self.current_trick)
            logging.info(f'Player {player.player_id} played {card_played.suit} {card_played.rank}')
            self.leading_suit = self.leading_suit or card_played.suit
            self.current_trick.append((player.player_id, card_played))
            self.current_player = (self.current_player + 1) % 4
        round_result, failure_message = self._resolve_winner(round_number)
        [player.update_possible_communications() for player in self.players if not player.has_communicated]
        return round_result

    def _resolve_winner(self, round_number: int) -> tuple[bool, str]:
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
                    return False, failure_message
        logging.info(f'Player {winning_player} won round {round_number}')
        self.current_player = winning_player
        return True, ""

    def get_state(self, player_id: int) -> dict:
        player: CrewPlayer = self.players[player_id]
        return {
            'hand': [card.to_tuple() for card in player.hand],
            'missions': [card.to_tuple() for card in player.missions],
            'signals': [(card.to_tuple(), signal.value) for card, signal in player.signals],
        }
    
    def get_num_players(self): return 4
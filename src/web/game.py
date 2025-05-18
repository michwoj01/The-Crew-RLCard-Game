import numpy as np
import logging
from utils.card import Communicate, CrewCard
from web.players import IntelligentCrewPlayer, HumanCrewPlayer, CrewPlayer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("game.log"),
        logging.StreamHandler()
    ]
)


class HumanCrewGame:
    def __init__(self):
        self.players: list[CrewPlayer] = []
        self.deck: list[CrewCard] = []
        self.tricks: list[list[tuple[int, CrewCard]]] = []
        self.tasks: list[tuple[int, CrewCard]] = []
        self.show_hands = False
        self.communication_log: list[tuple[int, Communicate]] = []
        self.starting_player = None
        self.leading_suit = None
        self.current_player = None
        self.current_trick: list[tuple[int, CrewCard]] = []
        self.winner = None

    async def init_game(self, players: list[CrewPlayer], no_tasks: int, show_hands: bool = False):
        self.players = players
        self.show_hands = show_hands
        self.deck = self._initialize_deck()
        self._deal_cards()
        self.starting_player = self._find_starting_player()
        await self._assign_tasks(no_tasks)

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

    async def _assign_tasks(self, no_of_tasks: int):
        if isinstance(self.players[0], HumanCrewPlayer):
            await self._send_human_player_state(0)
        task_cards = np.random.choice(
            [card for card in self.deck if not card.is_rocket], no_of_tasks, replace=False)
        for i in [(self.starting_player + j) % len(self.players) for j in range(no_of_tasks)]:
            chosen_task = await self.players[i].choose_task(task_cards) if isinstance(self.players[i], HumanCrewPlayer) \
                else self.players[i].choose_task(task_cards)
            self.tasks.append((i, chosen_task))
            self.players[i].tasks_assigned.append(chosen_task)
            task_cards = [card for card in task_cards if card != chosen_task]

    def _find_starting_player(self) -> int:
        starting_player = next((player.player_id for player in self.players if any(
            card.is_rocket and card.rank == 4 for card in player.hand)), 0)
        logging.info(f'Player {starting_player} has the 4 Rocket')
        return starting_player

    async def play_game(self):
        for i in range(10):
            if not await self._play_round(i):
                break
        else:
            if isinstance(self.players[0], HumanCrewPlayer):
                await self._send_human_player_game_won()

    async def _play_round(self, round_number: int) -> bool:
        self.current_trick = []
        self.leading_suit = None
        logging.info(f'Starting round {round_number}')
        if self.show_hands:
            [player.show_hand_and_task() for player in self.players]
        logging.info("Communication log: %s", [
                     (log[0], f"{log[1][0]}-{log[1][1]}") for log in self.communication_log])
        logging.info("Missions: %s", [(task[0], str(
            task[1])) for task in self.tasks])
        self.current_player = self.starting_player
        for _ in range(4):
            player: CrewPlayer = self.players[self.current_player]
            if not player.has_communicated:
                communicate = None
                if isinstance(player, HumanCrewPlayer):
                    await self._send_human_player_communication(round_number)
                    communicate = await player.communicate()
                else:
                    communicate = player.communicate()
                logging.info(
                    f'Player {player.player_id} communicated {communicate[1][0]} {communicate[1][1]}')
                self.communication_log.append(communicate)
            if isinstance(player, HumanCrewPlayer):
                await self._send_human_player_state(round_number)
                card_played = await player.play_card(self.current_trick)
            elif isinstance(player, IntelligentCrewPlayer):
                player.update_state(self.tasks)
                card_played = player.play_card(self.current_trick)
            else:
                card_played = player.play_card(self.current_trick)
            logging.info(
                f'Player {player.player_id} played {card_played.suit} {card_played.rank}')
            self.leading_suit = self.leading_suit or card_played.suit
            self.current_trick.append((player.player_id, card_played))
            self.current_player = (self.current_player + 1) % 4
        round_result, failure_message = self._resolve_winner(round_number)
        if not round_result and isinstance(self.players[0], HumanCrewPlayer):
            await self._send_human_player_game_lost(failure_message)
            return False
        [player.update_possible_communications()
         for player in self.players if not player.has_communicated]
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
        for owner, task in self.tasks:
            if task in played_cards:
                if owner == winning_player:
                    self.players[winning_player].complete_task(task)
                else:
                    failure_message = f'Player {winning_player} took {task}, but it should have been Player {owner}.'
                    logging.error(failure_message)
                    return False, failure_message
        logging.info(f'Player {winning_player} won round {round_number}')
        self.starting_player = winning_player
        return True, ""

    async def _send_human_player_communication(self, round_number: int):
        player = self.players[0]
        await player.websocket.send_json({
            "action": "round_state",
            "roundState": {
                "startingPlayer": self.starting_player,
                "roundNumber": round_number,
                "hand": [str(card) for card in player.hand],
                "tasks": [(p.player_id, [str(task) for task in p.tasks_assigned]) for p in self.players],
                "communications": [(log[0], f"{log[1][0]}-{log[1][1]}") for log in self.communication_log],
                "playedCards": [],
            }
        })

    async def _send_human_player_state(self, round_number: int):
        player = self.players[0]
        await player.websocket.send_json({
            "action": "round_state",
            "roundState": {
                "startingPlayer": self.starting_player,
                "roundNumber": round_number,
                "hand": [str(card) for card in player.hand],
                "tasks": [(p.player_id, [str(task) for task in p.tasks_assigned]) for p in self.players],
                "communications": [(log[0], f"{log[1][0]}-{log[1][1]}") for log in self.communication_log],
                "playedCards": [(p_id, str(card)) for p_id, card in self.current_trick],
            }
        })

    async def _send_human_player_game_won(self):
        await self.players[0].websocket.send_json({"action": "game_over"})

    async def _send_human_player_game_lost(self, failure_message: str):
        await self.players[0].websocket.send_json({"action": "round_failure", "message": failure_message})

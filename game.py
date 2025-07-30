from typing import List

from action_event import PlayCardAction, ChooseTaskAction, SignalAction, SkipSignalAction
from card import CrewCard, CrewTask
from dealer import Dealer
from move import PlayCardMove, CrewMove, ChooseTaskMove, SignalMove, SkipMove
from player import CrewPlayer


class Game:

    @property
    def game_phase(self) -> str:
        if len(self.dealer.tasks) > 0:
            result = 'choosing tasks'
        elif self.is_over():
            result = 'game over'
        elif not self.skip_signals and self.signaling_phase:
            result = 'signaling'
        else:
            result = 'playing card'
        return result

    def __init__(self, num_players: int, no_tasks: int, np_random,
                 skip_signals: bool = False, is_clone: bool = False, eval_mode: bool = False, eval_hand_id: int = 0):
        self.dealer = None
        self.no_tasks = no_tasks
        self.np_random = np_random
        self.num_players: int = num_players
        self.skip_signals: bool = skip_signals
        self.eval_mode = eval_mode
        self.eval_hand_id = eval_hand_id
        self.players: List[CrewPlayer] = []
        self.payoffs: List[List[float]] = []
        for player_id in range(num_players):
            self.players.append(CrewPlayer(player_id=player_id))
            self.payoffs.append([])
        self.current_player_id: int = 0
        self.starting_player_id: int = 0
        self.play_card_count: int = 0
        self.move_sheet: List[CrewMove] = []
        self.impossible_to_win: bool = False
        self.signal_counter: int = 0
        self.signaling_phase: bool = True
        self.tasks: List[CrewTask] = []
        self.trick_count: int = 0
        self.is_clone = is_clone

    def init_game(self):
        self.dealer: Dealer = Dealer(np_random=self.np_random, no_tasks=self.no_tasks,
                                     eval_mode=self.eval_mode, eval_hand_id=self.eval_hand_id)
        for player_id in range(self.num_players):
            player = self.players[player_id]
            self.dealer.deal_cards(player=player, num=10)
        self.starting_player_id = next(
            (player for player in self.players if any(
                card.suit == CrewCard.trump_suit and card.rank == 4 for card in player.hand)),
            None
        ).player_id
        self.current_player_id = self.starting_player_id

    def is_over(self) -> bool:
        card_over = True
        task_over = True
        for player in self.players:
            card_over = card_over and not player.hand
        for task in self.tasks:
            task_over = task_over and task.taken

        return self.impossible_to_win or (len(self.dealer.tasks) == 0 and (card_over or task_over))

    def get_current_player(self) -> CrewPlayer:
        return self.players[self.current_player_id]

    def get_current_player_id(self) -> int:
        return self.current_player_id

    def get_trick_moves(self) -> List[PlayCardMove]:
        trick_moves: List[PlayCardMove] = []
        counter = self.play_card_count
        if counter > 0:
            for move in self.move_sheet[-counter:]:
                if isinstance(move, PlayCardMove):
                    trick_moves.append(move)
            if len(trick_moves) != counter:
                raise Exception(
                    f'get_trick_moves: count of trick_moves={[str(move.card) for move in trick_moves]} does not equal {counter}')
        return trick_moves

    def play_card(self, action: PlayCardAction):
        current_player = self.players[self.current_player_id]
        self.move_sheet.append(PlayCardMove(current_player.player_id, action))
        card = action.card
        current_player.remove_card_from_hand(card=card)
        self.play_card_count += 1
        trick_moves = self.get_trick_moves()
        self.payoffs[self.current_player_id].append(0)
        if len(trick_moves) == 4:
            leading_card = trick_moves[0].card
            trick_winner = trick_moves[0].player_id
            for move in trick_moves[1:]:
                trick_card = move.card
                trick_player = move.player_id
                if trick_card.suit == leading_card.suit:
                    if trick_card.rank > leading_card.rank:
                        leading_card = trick_card
                        trick_winner = trick_player
                elif trick_card.suit == CrewCard.trump_suit:
                    leading_card = trick_card
                    trick_winner = trick_player
            self.starting_player_id = trick_winner
            self.current_player_id = self.starting_player_id
            self.check_tasks(trick_winner, [move.card for move in trick_moves])
            if not self.skip_signals and self.signal_counter < 4:
                self.signaling_phase = True
                while self.players[self.current_player_id].signal is not None:
                    self.current_player_id = (self.current_player_id + 1) % 4
            self.play_card_count = 0
            self.trick_count += 1
        else:
            self.current_player_id = (self.current_player_id + 1) % 4

    def signal(self, action: SignalAction | SkipSignalAction):
        current_player = self.players[self.current_player_id]
        if isinstance(action, SkipSignalAction):
            self.move_sheet.append(SkipMove(current_player.player_id, action))
        else:
            current_player.signal_card(action.card, action.signal_type)
            self.move_sheet.append(SignalMove(current_player.player_id, action))
            self.signal_counter += 1
        self.payoffs[self.current_player_id].append(0)
        if self.signal_counter == 4:
            self.signaling_phase = False
            self.current_player_id = self.starting_player_id
        else:
            for _ in range(4):
                self.current_player_id = (self.current_player_id + 1) % 4
                if self.current_player_id == self.starting_player_id:
                    self.signaling_phase = False
                    break
                if self.players[self.current_player_id].signal is None:
                    break
            else:
                raise Exception("All players have signals, but signaling phase is not over.")

    def choose_task(self, action: ChooseTaskAction):
        self.move_sheet.append(ChooseTaskMove(self.current_player_id, action))
        card = action.card
        self.tasks.append(self.dealer.assign_task(self.current_player_id, card))
        self.payoffs[self.current_player_id].append(0)
        if len(self.dealer.tasks) > 0:
            self.current_player_id = (self.current_player_id + 1) % 4
        else:
            self.current_player_id = self.starting_player_id

    def check_tasks(self, trick_winner: int, won_trick: list[CrewCard]):
        for task in self.tasks:
            if task.card in won_trick:
                task_completed = task.complete(taker=trick_winner)
                if not task_completed:
                    # if not self.is_clone:
                    #     print(
                    #         f'Task {task.card} should have been taken by {task.owner}, but was taken by {trick_winner}.')
                    self.impossible_to_win = True
                    break
                else:
                    for player in self.players:
                        self.payoffs[player.player_id][-1] = 0.5

    def clone(self) -> 'Game':
        new_game = Game(
            num_players=self.num_players,
            no_tasks=len(self.dealer.tasks) + len(self.tasks),
            np_random=None,
            skip_signals=self.skip_signals,
            is_clone=True,
            eval_mode=self.eval_mode,
            eval_hand_id=self.eval_hand_id
        )
        new_game.current_player_id = self.current_player_id
        new_game.starting_player_id = self.starting_player_id
        new_game.play_card_count = self.play_card_count
        new_game.move_sheet = [move for move in self.move_sheet]
        new_game.impossible_to_win = self.impossible_to_win
        new_game.signal_counter = self.signal_counter
        new_game.signaling_phase = self.signaling_phase
        new_game.trick_count = self.trick_count

        new_game.payoffs = [payoff_list.copy() for payoff_list in self.payoffs]
        new_game.players = [player.clone() for player in self.players]
        new_game.tasks = [task.clone() for task in self.tasks]
        new_game.dealer = self.dealer.clone()

        return new_game

from typing import List

from numpy.random import RandomState

from action_event import PlayCardAction, ChooseTaskAction, SignalAction, SkipSignalAction
from card import CrewCard, CrewTask
from dealer import Dealer
from player import CrewPlayer
from move import PlayCardMove, CrewMove, ChooseTaskMove, SignalMove, SkipMove


class Round:

    @property
    def round_phase(self) -> str:
        if len(self.dealer.tasks) > 0:
            result = 'choosing tasks'
        elif self.is_over():
            result = 'game over'
        elif self.signaling_phase:
            result = 'signaling'
        else:
            result = 'playing card'
        return result

    def __init__(self, num_players: int, np_random: RandomState):
        self.np_random: RandomState = np_random
        self.dealer: Dealer = Dealer(self.np_random)
        self.num_players: int = num_players
        self.players: List[CrewPlayer] = []
        for player_id in range(num_players):
            self.players.append(CrewPlayer(
                player_id=player_id, np_random=self.np_random))
        self.current_player_id: int = 0
        self.play_card_count: int = 0
        self.move_sheet: List[CrewMove] = []
        self.impossible_to_win: bool = False
        self.signal_counter: int = 0
        self.signaling_phase: bool = True
        self.tasks: List[CrewTask] = []
        self.trick_count: int = 1

    def init_round(self):
        for player_id in range(self.num_players):
            player = self.players[player_id]
            self.dealer.deal_cards(player=player, num=10)
        self.current_player_id = next(
            (player for player in self.players if any(
                card.suit == 'R' and card.rank == 4 for card in player.hand)),
            None
        ).player_id
        self.dealer.prepare_tasks()

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

        if len(trick_moves) == 4:
            trump_suit = CrewCard.suits[4]
            leading_card = trick_moves[0].card
            trick_winner = trick_moves[0].player_id
            for move in trick_moves[1:]:
                trick_card = move.card
                trick_player = move.player_id
                if trick_card.suit == leading_card.suit:
                    if trick_card.rank > leading_card.rank:
                        leading_card = trick_card
                        trick_winner = trick_player
                elif trick_card.suit == trump_suit:
                    leading_card = trick_card
                    trick_winner = trick_player
            self.current_player_id = trick_winner
            self.check_tasks(trick_winner, [move.card for move in trick_moves])
            self.signaling_phase = True
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
            self.move_sheet.append(SignalMove(
                current_player.player_id, action))
        self.signal_counter += 1
        if self.signal_counter == 4:
            self.signaling_phase = False
            self.signal_counter = 0
        self.current_player_id = (self.current_player_id + 1) % 4

    def choose_task(self, action: ChooseTaskAction):
        self.move_sheet.append(ChooseTaskMove(self.current_player_id, action))
        card = action.card
        self.tasks.append(self.dealer.assign_task(
            self.current_player_id, card))
        self.current_player_id = (self.current_player_id + 1) % 4

    def check_tasks(self, trick_winner: int, won_trick: list[CrewCard]):
        for task in self.tasks:
            if task.card in won_trick:
                task_completed = task.complete(taker=trick_winner)
                if not task_completed:
                    self.impossible_to_win = True
                    break

from typing import List

from numpy.random import RandomState

from action_event import PlayCardAction, ChooseTaskAction, SignalAction, SkipSignalAction
from card import CrewCard
from dealer import Dealer
from player import CrewPlayer
from move import PlayCardMove, CrewMove, DealHandMove, ChooseTaskMove, SignalMove, SkipMove


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
        dealer_id = 1
        self.dealer: Dealer = Dealer(self.np_random)
        self.players: List[CrewPlayer] = []
        for player_id in range(num_players):
            self.players.append(CrewPlayer(
                player_id=player_id, np_random=self.np_random))
        self.current_player_id: int = dealer_id
        self.play_card_count: int = 0
        self.move_sheet: List[CrewMove] = []
        self.move_sheet.append(DealHandMove(dealer=self.players[dealer_id], shuffled_deck=self.dealer.shuffled_deck))
        self.impossible_to_win: bool = False
        self.signal_counter: int = 0
        self.signaling_phase: bool = True

    def is_over(self) -> bool:
        card_over = True
        task_over = True
        for player in self.players:
            card_over = card_over and not player.hand
            task_over = task_over and not player.tasks_assigned

        return self.impossible_to_win or (len(self.dealer.tasks) == 0 and (card_over or task_over))

    def get_current_player(self) -> CrewPlayer:
        return self.players[self.current_player_id]

    def get_trick_moves(self) -> List[PlayCardMove]:
        trick_moves: List[PlayCardMove] = []
        if self.play_card_count > 0:
            trick_pile_count = self.play_card_count % 4
            if trick_pile_count == 0:
                trick_pile_count = 4
            for move in self.move_sheet[-trick_pile_count:]:
                if isinstance(move, PlayCardMove):
                    trick_moves.append(move)
            if len(trick_moves) != trick_pile_count:
                raise Exception(
                    f'get_trick_moves: count of trick_moves={[str(move.card) for move in trick_moves]} does not equal {trick_pile_count}')
        return trick_moves

    def play_card(self, action: PlayCardAction):
        current_player = self.players[self.current_player_id]
        self.move_sheet.append(PlayCardMove(current_player, action))
        card = action.card
        current_player.remove_card_from_hand(card=card)
        self.play_card_count += 1
        trick_moves = self.get_trick_moves()

        if len(trick_moves) == 4:
            trump_suit = CrewCard.suits[4]
            leading_card = trick_moves[0].card
            trick_winner = trick_moves[0].player
            for move in trick_moves[1:]:
                trick_card = move.card
                trick_player = move.player
                if trick_card.suit == leading_card.suit:
                    if trick_card.card_id > leading_card.card_id:
                        leading_card = trick_card
                        trick_winner = trick_player
                elif trick_card.suit == trump_suit:
                    leading_card = trick_card
                    trick_winner = trick_player
            self.current_player_id = trick_winner.player_id
            trick_winner.complete_task(card_moves=trick_moves)
            self.check_tasks(trick_winner, [move.card for move in trick_moves])
            self.signaling_phase = True
        else:
            self.current_player_id = (self.current_player_id + 1) % 4

    def signal(self, action: SignalAction | SkipSignalAction):
        current_player = self.players[self.current_player_id]
        if isinstance(action, SkipSignalAction):
            self.move_sheet.append(SkipMove(current_player, action))
        else:
            current_player.signal_card(action.card, action.signal_type)
            self.move_sheet.append(SignalMove(current_player, action))
        self.signal_counter += 1
        if self.signal_counter == 4:
            self.signaling_phase = False
            self.signal_counter = 0
        self.current_player_id = (self.current_player_id + 1) % 4

    def choose_task(self, action: ChooseTaskAction):
        current_player = self.players[self.current_player_id]
        self.move_sheet.append(ChooseTaskMove(current_player, action))
        task = action.task
        self.dealer.assign_task(current_player, task)
        self.current_player_id = (self.current_player_id + 1) % 4

    def check_tasks(self, trick_winner: CrewPlayer, won_trick: list[CrewCard]):
        for player in self.players:
            if any(item in won_trick for item in player.tasks_assigned) and player.player_id != trick_winner.player_id:
                self.impossible_to_win = True
                break

    # def get_perfect_information(self):
    #     state = {}
    #     trick_moves = [None, None, None, None]
    #     for trick_move in self.get_trick_moves():
    #         trick_moves[trick_move.player.player_id] = trick_move.card
    #     state['move_count'] = len(self.move_sheet)
    #     state['current_player_id'] = self.current_player_id
    #     state['round_phase'] = self.round_phase
    #     state['hands'] = [player.hand for player in self.players]
    #     state['tasks'] = [player.tasks_assigned for player in self.players]
    #     state['trick_moves'] = trick_moves
    #     return state
    #
    # def print_scene(self):
    #     print(
    #         f'===== Move: {len(self.move_sheet)} player: {self.players[self.current_player_id]} phase: {self.round_phase} =====')
    #     for player in self.players:
    #         print(f'{player}: {[str(card) for card in player.hand]}')
    #     trick_pile = ['None', 'None', 'None', 'None']
    #     for trick_move in self.get_trick_moves():
    #         trick_pile[trick_move.player.player_id] = trick_move.card
    #     print(f'trick_pile: {[str(card) for card in trick_pile]}')

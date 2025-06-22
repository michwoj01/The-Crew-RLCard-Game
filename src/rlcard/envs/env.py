import numpy as np

from src.bridge.action_event import ActionEvent
from src.rlcard.utils import seeding


class Env(object):

    def __init__(self, game, config):
        self.game = game
        self.allow_step_back = self.game.allow_step_back = config['allow_step_back']
        self.action_recorder = []

        # Get the number of players/actions in this game
        self.num_players = self.game.get_num_players()
        self.num_actions = self.game.get_num_actions()

        # A counter for the timesteps
        self.timestep = 0

        # Set random seed, default is None
        self.seed(config['seed'])

    def reset(self):
        state, player_id = self.game.init_game()
        self.action_recorder = []
        return self._extract_state(state), player_id

    def step(self, action_id: int):
        action = self._decode_action(action_id)

        self.timestep += 1
        self.action_recorder.append((self.get_player_id(), action))
        next_state, player_id = self.game.step(action)

        return self._extract_state(next_state), player_id

    def step_back(self):
        if not self.allow_step_back:
            raise Exception('Step back is off. To use step_back, please set allow_step_back=True in rlcard.make')

        if not self.game.step_back():
            return False

        player_id = self.get_player_id()
        state = self.get_state(player_id)

        return state, player_id

    def set_agents(self, agents):
        self.agents = agents

    def run(self, is_training=False):
        trajectories = [[] for _ in range(self.num_players)]
        state, player_id = self.reset()

        # Loop to play the game
        trajectories[player_id].append(state)
        while not self.is_over():
            # Agent plays
            if not is_training:
                action_id, _ = self.agents[player_id].eval_step(state)
            else:
                action_id = self.agents[player_id].step(state)

            # Environment steps
            next_state, next_player_id = self.step(action_id)
            # Save action
            trajectories[player_id].append(action_id)

            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.game.is_over():
                trajectories[player_id].append(state)
        # Add a final state to all the players
        for player_id in range(self.num_players):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()

        return trajectories, payoffs

    def is_over(self):
        return self.game.is_over()

    def get_player_id(self):
        return self.game.get_player_id()

    def get_state(self, player_id):
        return self._extract_state(self.game.get_state(player_id))

    def get_payoffs(self):
        raise NotImplementedError

    def get_perfect_information(self):
        raise NotImplementedError

    def get_action_feature(self, action):
        # By default, we use one-hot encoding
        feature = np.zeros(self.num_actions, dtype=np.int8)
        feature[action] = 1
        return feature

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.game.np_random = self.np_random
        return seed

    def _extract_state(self, state):
        raise NotImplementedError

    def _decode_action(self, action_id: int) -> ActionEvent:
        raise NotImplementedError

    def _get_legal_actions(self):
        raise NotImplementedError

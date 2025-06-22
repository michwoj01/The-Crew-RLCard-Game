import numpy as np


class RandomAgent(object):

    def __init__(self, num_actions):
        self.use_raw = False
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        return np.random.choice(state['legal_actions'])

    def eval_step(self, state):
        legal_actions = state['legal_actions']
        probs = [0 for _ in range(self.num_actions)]
        for i in legal_actions:
            probs[i] = 1 / len(legal_actions)
        info = {'probs': {legal_actions[i]: probs[legal_actions[i]] for i in
                          range(len(legal_actions))}}

        return self.step(state), info

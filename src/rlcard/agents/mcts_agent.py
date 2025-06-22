import copy
import math
import random

class TreeNode:
    def __init__(self, parent, prior_prob):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.prior = prior_prob

    def expand(self, action_priors):
        for action_id, prob in action_priors:
            if action_id not in self.children:
                self.children[action_id] = TreeNode(self, prob)

    def select(self, c_puct):
        best_action, best_node = None, None
        best_score = -float('inf')
        for action_id, child in self.children.items():
            u = (c_puct * child.prior *
                 math.sqrt(self.visits + 1e-8) / (1 + child.visits))
            q = child.value / (child.visits + 1e-8)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action_id
                best_node = child
        return best_action, best_node

    def update(self, leaf_value):
        self.visits += 1
        self.value += leaf_value

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

class MCTS:
    def __init__(self, env, c_puct=1.4, n_simulations=100):
        self.root = TreeNode(parent=None, prior_prob=1.0)
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.env = env

    def _simulate(self, env_copy):
        while not env_copy.game.is_over():
            legal_actions = env_copy.get_legal_actions()
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            env_copy.step(action)
        payoffs = env_copy.get_payoffs()
        return payoffs

    def _rollout(self, env_copy, node):
        if node.is_leaf():
            payoffs = self._simulate(env_copy)
            return payoffs[self.env.get_player_id()]
        else:
            action_id, next_node = node.select(self.c_puct)
            env_copy.step(action_id)
            leaf_value = self._rollout(env_copy, next_node)
            next_node.update_recursive(leaf_value)
            return leaf_value

    def run(self, state):
        self.root = TreeNode(parent=None, prior_prob=1.0)
        for _ in range(self.n_simulations):
            env_copy = copy.deepcopy(self.env)
            self._rollout(env_copy, self.root)

        visits = [(act_id, node.visits) for act_id, node in self.root.children.items()]
        if not visits:
            legal_actions = state['legal_actions']
            return random.choice(legal_actions)
        visits.sort(key=lambda x: x[1], reverse=True)
        best_action_id = visits[0][0]
        return best_action_id

    def update_with_action(self, last_action_id):
        if last_action_id in self.root.children:
            self.root = self.root.children[last_action_id]
            self.root.parent = None
        else:
            self.root = TreeNode(parent=None, prior_prob=1.0)

class MCTSAgent:
    def __init__(self, env, n_simulations=100, c_puct=1.4):
        self.env = env
        self.mcts = MCTS(env, c_puct=c_puct, n_simulations=n_simulations)
        self.use_raw = False

    def step(self, state):
        legal_actions = state.get('legal_actions', state.get('raw_legal_actions'))
        if len(legal_actions) == 1:
            return legal_actions[0]
        action_id = self.mcts.run(state)
        return action_id

    def eval_step(self, state):
        return self.step(state), None
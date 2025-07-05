import copy
import random

import math


class TreeNode:
    def __init__(self, parent, env):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.env = env

    def expand(self, env):
        legal_actions = env.get_legal_actions()
        unexplored_actions = list(set(legal_actions) - set(self.children))
        action_id = random.choice(unexplored_actions)
        new_env = copy.deepcopy(env)
        new_env.step(action_id)
        self.children[action_id] = TreeNode(self, new_env)
        return self.children[action_id]

    def update(self, leaf_value):
        self.visits += 1
        self.value += leaf_value

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def simulate(self):
        simulation_env_copy = copy.deepcopy(self.env)
        while not simulation_env_copy.game.is_over():
            legal_actions = simulation_env_copy.get_legal_actions()
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            simulation_env_copy.step(action)
        payoffs = simulation_env_copy.get_payoffs()
        return sum(payoffs[self.env.get_player_id()][:-1])


class MCTS:
    def __init__(self, env, n_simulations=100):
        self.n_simulations = n_simulations
        self.env = env
        self.all_visits = 0

    def run(self, state):
        env_copy = copy.deepcopy(self.env)
        root = TreeNode(parent=None, env=env_copy)
        extendable_leafs = [root]
        for _ in range(self.n_simulations):
            best_node = None
            best_score = -float('inf')
            if self.all_visits == 0:
                best_node = root
            else:
                for leaf in extendable_leafs:
                    exploration_factor = math.sqrt((2 * math.log(self.all_visits, math.e)) / leaf.visits)
                    score = leaf.value + exploration_factor
                    if score > best_score:
                        best_score = score
                        best_node = leaf

            new_node = best_node.expand(best_node.env)
            leaf_value = new_node.simulate()
            new_node.update_recursive(leaf_value)
            self.all_visits += 1

            if not new_node.env.game.is_over():
                extendable_leafs.append(new_node)
            if len(list(set(best_node.env.get_legal_actions()) - set(best_node.children))) == 0:
                extendable_leafs.remove(best_node)

        visits = [(act_id, node.visits) for act_id, node in root.children.items()]
        if not visits:
            legal_actions = state['legal_actions']
            return random.choice(legal_actions)
        visits.sort(key=lambda x: x[1], reverse=True)
        best_action_id = visits[0][0]
        return best_action_id


class MCTSAgent:
    def __init__(self, env, n_simulations=100):
        self.env = env
        self.mcts = MCTS(env, n_simulations=n_simulations)
        self.use_raw = False

    def step(self, state):
        legal_actions = state['legal_actions']
        if len(legal_actions) == 1:
            return legal_actions[0]
        action_id = self.mcts.run(state)
        return action_id

    def eval_step(self, state):
        return self.step(state), None

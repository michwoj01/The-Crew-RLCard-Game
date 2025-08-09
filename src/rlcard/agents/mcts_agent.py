import random

import math

from src.rlcard.utils import reorganize


class TreeNode:
    def __init__(self, parent, env, action_taken=None):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.env = env
        self.action_taken = action_taken  # Action that led to this node
        self.player_id = env.get_player_id()  # Player who will act from this state

    def is_fully_expanded(self):
        legal_actions = self.env._get_legal_actions()
        return len(self.children) == len(legal_actions)

    def expand(self):
        legal_actions = self.env._get_legal_actions()
        unexplored_actions = list(set(legal_actions) - set(self.children.keys()))
        if not unexplored_actions:
            return None

        action_id = random.choice(unexplored_actions)
        new_env = self.env.clone()
        new_env.step(action_id)
        child = TreeNode(self, new_env, action_id)
        self.children[action_id] = child
        return child

    def select_best_child(self, c_param=1.414):
        """Select child using UCB1 formula"""
        if not self.children:
            return None

        best_score = -float('inf')
        best_child = None

        for child in self.children.values():
            if child.visits == 0:
                return child  # Prioritize unvisited children

            # UCB1 formula
            exploitation = child.value / child.visits
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            score = exploitation + exploration

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def update(self, result):
        self.visits += 1
        self.value += result

    def backpropagate(self, result, original_player):
        # In The Crew, all players win or lose together (cooperative game)
        # So we can use the result directly for all players
        self.update(result)
        if self.parent:
            self.parent.backpropagate(result, original_player)

    def simulate(self):
        simulation_env = self.env.clone()
        original_player = simulation_env.get_player_id()

        while not simulation_env.is_over():
            legal_actions = simulation_env._get_legal_actions()
            if not legal_actions:
                break
            action = random.choice(legal_actions)
            simulation_env.step(action)

        payoffs = simulation_env.get_payoffs()
        return payoffs[original_player]

    def simulate_with_dqn(self, agent):
        simulation_env = self.env.clone()
        original_player = simulation_env.get_player_id()

        agents = [agent for _ in range(simulation_env.num_players)]
        simulation_env.set_agents(agents)
        simulation_env.algorithm = 'dqn'

        trajectories, payoffs = simulation_env.run_without_reset(original_player, is_training=True)
        trajectories = reorganize(trajectories, payoffs)

        for ts in trajectories[0]:
            agents[0].feed(ts)

        return sum(simulation_env.get_payoffs()[original_player])


class MCTS:
    def __init__(self, env, dqn_agent, n_simulations=100, c_param=1.414):
        self.n_simulations = n_simulations
        self.env = env
        self.c_param = c_param
        self.dqn_agent = dqn_agent

    def run(self, state):
        """Run MCTS and return best action"""
        # Create root node from current environment state
        root_env = self.env.clone()
        root = TreeNode(parent=None, env=root_env)
        original_player = root_env.get_player_id()

        # If only one legal action, return it immediately
        legal_actions = state['legal_actions']
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else None

        # Run MCTS simulations
        for _ in range(self.n_simulations):
            # Selection: traverse tree using UCB1
            node = root
            path = [node]

            while not node.env.is_over() and node.is_fully_expanded():
                node = node.select_best_child(self.c_param)
                if node is None:
                    break
                path.append(node)

            # Expansion: add new child if possible
            if not node.env.is_over() and not node.is_fully_expanded():
                child = node.expand()
                if child:
                    node = child
                    path.append(node)

            # Simulation: random rollout from this node
            # result = node.simulate()  # RANDOM
            result = node.simulate_with_dqn(self.dqn_agent)  # DQN

            # Backpropagation: update all nodes in path
            node.backpropagate(result, original_player)

        # Select best action based on visit counts (most robust)
        if not root.children:
            return random.choice(legal_actions)

        best_action = max(root.children.keys(),
                          key=lambda a: root.children[a].visits)
        return best_action

    def get_action_stats(self, state):
        """Get statistics for all actions (useful for debugging)"""
        root_env = self.env.clone()
        root = TreeNode(parent=None, env=root_env)

        for _ in range(self.n_simulations):
            node = root
            while not node.env.is_over() and node.is_fully_expanded():
                node = node.select_best_child(self.c_param)
                if node is None:
                    break

            if not node.env.is_over() and not node.is_fully_expanded():
                child = node.expand()
                if child:
                    node = child

            # result = node.simulate()  # RANDOM
            result = node.simulate_with_dqn(self.dqn_agent)  # DQN
            node.backpropagate(result, root_env.get_player_id())

        stats = {}
        for action_id, child in root.children.items():
            stats[action_id] = {
                'visits': child.visits,
                'value': child.value,
                'avg_value': child.value / child.visits if child.visits > 0 else 0
            }
        return stats


class MCTSAgent:
    def __init__(self, env, dqn_agent, n_simulations=100, c_param=1.414):
        self.env = env
        self.n_simulations = n_simulations
        self.c_param = c_param
        self.use_raw = False
        self.dqn_agent = dqn_agent

    def step(self, state):
        legal_actions = state['legal_actions']
        if len(legal_actions) <= 1:
            return legal_actions[0] if legal_actions else None

        mcts = MCTS(self.env, self.dqn_agent, self.n_simulations, self.c_param)
        action_id = mcts.run(state)
        return action_id

    def eval_step(self, state):
        return self.step(state), None

    def set_simulations(self, n_simulations):
        self.n_simulations = n_simulations

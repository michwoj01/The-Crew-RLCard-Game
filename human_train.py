import random
from datetime import datetime

import numpy as np

from human_gui_agent import HumanAgentGUI
from mcts_agent import MCTSAgent
from registration import make, register


def run_game():
    register(
        env_id='crew',
        entry_point='env:CrewEnv',
    )

    env = make('crew', config={
        'num_players': 4,
        'no_tasks': 4,
        'skip_signals': False,
        'algorithm': 'mcts',
        'is_clone': False,
    })

    np.random.seed(42)
    random.seed(42)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"logs/crew_game_{timestamp}.txt"

    human_agent = HumanAgentGUI(log_file_path=log_file_path)
    agents = [human_agent,
              MCTSAgent(env, n_simulations=700),
              MCTSAgent(env, n_simulations=700),
              MCTSAgent(env, n_simulations=1000)]

    env.set_agents(agents)

    trajectories, payoffs = env.run(is_training=False)

    team_won = all(p > 0 for p in payoffs)
    human_agent.log_game_result(team_won)

if __name__ == '__main__':
    run_game()

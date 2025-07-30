import random
import sys
from datetime import datetime

import numpy as np

from src.rlcard.agents import HumanAgentGUI, MCTSAgent
from src.rlcard.envs import register, make


def run_game(no_tasks):
    register(
        env_id='crew',
        entry_point='src.rlcard.envs:CrewEnv',
    )

    env = make('crew', config={
        'num_players': 4,
        'no_tasks': no_tasks,
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
              MCTSAgent(env, n_simulations=1000),
              MCTSAgent(env, n_simulations=700),
              MCTSAgent(env, n_simulations=1000)]

    env.set_agents(agents)

    trajectories, payoffs = env.run(is_training=False)

    team_won = all(p > 0 for p in payoffs)
    human_agent.log_game_result(team_won)

    if team_won:
        print("🎉 Team won! All tasks completed successfully!")
    else:
        print("😞 Team lost. Tasks were not completed correctly.")


if __name__ == '__main__':
    run_game(int(sys.argv[1]))
